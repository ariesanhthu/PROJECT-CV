#!/usr/bin/env python3
"""Audit merge-candidate classes with visual examples and texture similarity.

This script does two things for classes that are being merged by mapping:
1) Extracts representative object examples per source class from annotations.
2) Builds texture-aware similarity reports per merge target using a precomputed
   class-level texture similarity matrix.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge texture audit for FoodSeg103.")
    parser.add_argument(
        "--mapping-csv",
        type=Path,
        default=Path("test/foodseg103_rebalanced/class_mapping.csv"),
        help="Path to class_mapping.csv",
    )
    parser.add_argument(
        "--ann-dir",
        type=Path,
        default=Path("test/foodseg103/train/ann"),
        help="Directory containing Supervisely JSON annotations",
    )
    parser.add_argument(
        "--img-dir",
        type=Path,
        default=Path("test/foodseg103/train/img"),
        help="Directory containing paired RGB images",
    )
    parser.add_argument(
        "--texture-sim-csv",
        type=Path,
        default=Path("source/analyze/outputs/texture_full_colab/texture_similarity_matrix.csv"),
        help="Class texture similarity matrix CSV",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("source/analyze/outputs/merge_texture_audit"),
        help="Output directory",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=6,
        help="Number of visual samples to export per source class",
    )
    parser.add_argument(
        "--scan-limit",
        type=int,
        default=0,
        help="Max annotation files to scan (0 means no limit)",
    )
    return parser.parse_args()


def sanitize_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return safe.strip("_") or "class"


def to_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def decode_bitmap_mask(data_b64: str) -> np.ndarray:
    raw = base64.b64decode(data_b64, validate=True)
    try:
        im = Image.open(io.BytesIO(raw))
        im.load()
    except Exception:
        raw = zlib.decompress(raw)
        im = Image.open(io.BytesIO(raw))
        im.load()
    gray = np.array(im.convert("L"), dtype=np.uint8)
    return (gray > 0).astype(np.uint8)


def load_merge_groups(mapping_csv: Path) -> tuple[dict[str, list[str]], pd.DataFrame]:
    df = pd.read_csv(mapping_csv)
    for col in ("source_class", "target_class", "kept"):
        if col not in df.columns:
            raise KeyError(f"Missing required column in mapping: {col}")

    df["source_class"] = df["source_class"].astype(str).str.strip()
    df["target_class"] = df["target_class"].astype(str).str.strip()
    df = df[df["source_class"] != ""]

    # Merge candidates are rows where source label is mapped to a different target label.
    merge_rows = df[df["source_class"] != df["target_class"]].copy()

    grouped = merge_rows.groupby("target_class", sort=True)
    merge_groups: dict[str, list[str]] = {}
    for target, g in grouped:
        sources = sorted(g["source_class"].dropna().unique().tolist())
        if len(sources) > 1:
            merge_groups[target] = sources

    return merge_groups, merge_rows


def extract_object_crop(
    rgb: np.ndarray,
    patch_mask: np.ndarray,
    ox: int,
    oy: int,
    pad: int = 6,
) -> np.ndarray | None:
    ys, xs = np.where(patch_mask > 0)
    if ys.size == 0:
        return None

    x0 = max(0, ox + int(xs.min()) - pad)
    y0 = max(0, oy + int(ys.min()) - pad)
    x1 = min(rgb.shape[1], ox + int(xs.max()) + 1 + pad)
    y1 = min(rgb.shape[0], oy + int(ys.max()) + 1 + pad)
    if x1 <= x0 or y1 <= y0:
        return None

    crop = rgb[y0:y1, x0:x1].copy()

    py0 = max(0, y0 - oy)
    px0 = max(0, x0 - ox)
    py1 = min(patch_mask.shape[0], y1 - oy)
    px1 = min(patch_mask.shape[1], x1 - ox)

    soft_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
    soft_mask[(py0 - (y0 - oy)) : (py1 - (y0 - oy)), (px0 - (x0 - ox)) : (px1 - (x0 - ox))] = patch_mask[
        py0:py1, px0:px1
    ].astype(np.float32)

    # Darken the background inside the crop to make object texture easier to inspect.
    alpha = 0.28 + 0.72 * soft_mask
    out = np.clip(crop.astype(np.float32) * alpha[..., None], 0, 255).astype(np.uint8)
    return out


def collect_examples(
    ann_dir: Path,
    img_dir: Path,
    needed_classes: set[str],
    out_dir: Path,
    samples_per_class: int,
    scan_limit: int,
) -> pd.DataFrame:
    examples_dir = out_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    per_class_count: dict[str, int] = {c: 0 for c in sorted(needed_classes)}
    rows: list[dict[str, Any]] = []

    ann_files = sorted(ann_dir.glob("*.json"))
    if scan_limit > 0:
        ann_files = ann_files[:scan_limit]

    for ann_path in ann_files:
        if all(v >= samples_per_class for v in per_class_count.values()):
            break

        with ann_path.open("r", encoding="utf-8") as f:
            ann = json.load(f)

        objects = ann.get("objects", [])
        target_objects = [obj for obj in objects if str(obj.get("classTitle", "")).strip() in needed_classes]
        if not target_objects:
            continue

        stem = ann_path.name.removesuffix(".json")
        img_path = img_dir / stem
        if not img_path.exists():
            continue

        rgb = np.array(Image.open(img_path).convert("RGB"))

        for obj_idx, obj in enumerate(target_objects):
            class_name = str(obj.get("classTitle", "")).strip()
            if class_name not in needed_classes:
                continue
            if per_class_count[class_name] >= samples_per_class:
                continue

            bitmap = obj.get("bitmap") or {}
            data_b64 = bitmap.get("data")
            origin = bitmap.get("origin", [0, 0])
            if not data_b64:
                continue

            try:
                patch_mask = decode_bitmap_mask(data_b64)
            except Exception:
                continue

            ox, oy = int(origin[0]), int(origin[1])
            crop = extract_object_crop(rgb, patch_mask, ox=ox, oy=oy)
            if crop is None:
                continue

            class_dir = examples_dir / sanitize_name(class_name)
            class_dir.mkdir(parents=True, exist_ok=True)

            idx = per_class_count[class_name]
            out_name = f"{idx:02d}_{stem}_{obj_idx:02d}.png"
            out_path = class_dir / out_name
            Image.fromarray(crop).save(out_path)

            per_class_count[class_name] += 1
            rows.append(
                {
                    "class_name": class_name,
                    "sample_index": idx,
                    "ann_file": ann_path.name,
                    "image_file": img_path.name,
                    "output_path": str(out_path.as_posix()),
                }
            )

    return pd.DataFrame(rows)


def render_group_gallery(
    target: str,
    sources: list[str],
    examples_df: pd.DataFrame,
    out_dir: Path,
    samples_per_class: int,
) -> Path:
    fig, axes = plt.subplots(
        nrows=len(sources),
        ncols=samples_per_class,
        figsize=(2.0 * samples_per_class, 2.2 * max(1, len(sources))),
        squeeze=False,
    )

    for r, src in enumerate(sources):
        class_rows = examples_df[examples_df["class_name"] == src].sort_values("sample_index")
        image_paths = class_rows["output_path"].tolist()

        for c in range(samples_per_class):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(image_paths):
                img = np.array(Image.open(image_paths[c]).convert("RGB"))
                ax.imshow(img)
            if c == 0:
                ax.set_title(src, fontsize=9, loc="left")

    fig.suptitle(f"Merge group: {target}", fontsize=12)
    fig.tight_layout()

    gallery_dir = out_dir / "galleries"
    gallery_dir.mkdir(parents=True, exist_ok=True)
    out_path = gallery_dir / f"group_{sanitize_name(target)}_gallery.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def compute_group_texture_similarity(
    texture_sim_csv: Path,
    merge_groups: dict[str, list[str]],
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sim_df = pd.read_csv(texture_sim_csv)
    if sim_df.empty:
        raise ValueError("Texture similarity CSV is empty.")

    class_col = sim_df.columns[0]
    sim_df = sim_df.set_index(class_col)

    all_pairs: list[dict[str, Any]] = []
    group_summary: list[dict[str, Any]] = []

    sim_out_dir = out_dir / "similarity"
    sim_out_dir.mkdir(parents=True, exist_ok=True)

    for target, sources in merge_groups.items():
        available = [s for s in sources if s in sim_df.index and s in sim_df.columns]
        missing = [s for s in sources if s not in available]

        if len(available) < 2:
            group_summary.append(
                {
                    "target_class": target,
                    "n_sources": len(sources),
                    "n_available_in_texture_matrix": len(available),
                    "missing_classes": "|".join(missing),
                    "mean_texture_similarity": np.nan,
                    "max_texture_similarity": np.nan,
                    "min_texture_similarity": np.nan,
                }
            )
            continue

        sub = sim_df.loc[available, available].astype(float)
        sub_path = sim_out_dir / f"group_{sanitize_name(target)}_matrix.csv"
        sub.to_csv(sub_path, index=True)

        vals = []
        for i in range(len(available)):
            for j in range(i + 1, len(available)):
                a = available[i]
                b = available[j]
                sim = float(sub.loc[a, b])
                vals.append(sim)
                all_pairs.append(
                    {
                        "target_class": target,
                        "class_a": a,
                        "class_b": b,
                        "texture_similarity": sim,
                        "texture_distance": 1.0 - sim,
                    }
                )

        group_summary.append(
            {
                "target_class": target,
                "n_sources": len(sources),
                "n_available_in_texture_matrix": len(available),
                "missing_classes": "|".join(missing),
                "mean_texture_similarity": float(np.mean(vals)),
                "max_texture_similarity": float(np.max(vals)),
                "min_texture_similarity": float(np.min(vals)),
            }
        )

        # Heatmap for quick merge inspection.
        fig, ax = plt.subplots(figsize=(1.0 + 0.8 * len(available), 1.0 + 0.8 * len(available)))
        im = ax.imshow(sub.values, vmin=-1.0, vmax=1.0, cmap="coolwarm")
        ax.set_xticks(range(len(available)))
        ax.set_yticks(range(len(available)))
        ax.set_xticklabels(available, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(available, fontsize=8)
        ax.set_title(f"Texture similarity: {target}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(sim_out_dir / f"group_{sanitize_name(target)}_heatmap.png", dpi=170)
        plt.close(fig)

    pairs_df = pd.DataFrame(all_pairs).sort_values("texture_similarity", ascending=False)
    summary_df = pd.DataFrame(group_summary).sort_values("target_class")
    return pairs_df, summary_df


def main() -> None:
    args = parse_args()

    merge_groups, merge_rows = load_merge_groups(args.mapping_csv)
    if not merge_groups:
        raise RuntimeError("No merge groups found (source_class -> target_class).")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    needed_classes = set()
    for _, srcs in merge_groups.items():
        needed_classes.update(srcs)

    examples_df = collect_examples(
        ann_dir=args.ann_dir,
        img_dir=args.img_dir,
        needed_classes=needed_classes,
        out_dir=args.out_dir,
        samples_per_class=args.samples_per_class,
        scan_limit=args.scan_limit,
    )

    example_csv = args.out_dir / "example_samples.csv"
    examples_df.to_csv(example_csv, index=False)

    galleries = []
    for target, sources in sorted(merge_groups.items()):
        gallery_path = render_group_gallery(
            target=target,
            sources=sources,
            examples_df=examples_df,
            out_dir=args.out_dir,
            samples_per_class=args.samples_per_class,
        )
        galleries.append({"target_class": target, "gallery_path": str(gallery_path.as_posix())})

    galleries_df = pd.DataFrame(galleries)
    galleries_df.to_csv(args.out_dir / "group_galleries.csv", index=False)

    pair_df, summary_df = compute_group_texture_similarity(
        texture_sim_csv=args.texture_sim_csv,
        merge_groups=merge_groups,
        out_dir=args.out_dir,
    )

    pair_df.to_csv(args.out_dir / "merge_texture_pairwise.csv", index=False)
    summary_df.to_csv(args.out_dir / "merge_group_texture_summary.csv", index=False)

    group_meta_rows = []
    for target, srcs in sorted(merge_groups.items()):
        g = merge_rows[merge_rows["target_class"] == target]
        group_meta_rows.append(
            {
                "target_class": target,
                "source_classes": "|".join(srcs),
                "n_source_classes": len(srcs),
                "kept_true_count": int(g["kept"].apply(to_bool).sum()),
                "kept_false_count": int((~g["kept"].apply(to_bool)).sum()),
            }
        )
    pd.DataFrame(group_meta_rows).to_csv(args.out_dir / "merge_groups.csv", index=False)

    print("Saved merge texture audit outputs to:", args.out_dir)
    print("-", (args.out_dir / "merge_groups.csv").as_posix())
    print("-", (args.out_dir / "example_samples.csv").as_posix())
    print("-", (args.out_dir / "group_galleries.csv").as_posix())
    print("-", (args.out_dir / "merge_texture_pairwise.csv").as_posix())
    print("-", (args.out_dir / "merge_group_texture_summary.csv").as_posix())


if __name__ == "__main__":
    main()
