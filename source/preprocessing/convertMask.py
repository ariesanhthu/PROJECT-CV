#!/usr/bin/env python3
"""Convert FoodSeg103 Supervisely JSON annotations to single-channel PNG masks."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import sys
import zlib
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# 103 ingredient classes → ids 0..102; unlabeled pixels → 103 (background)
BACKGROUND_ID = 103
# Rebalanced taxonomy: foreground ids 1..76; unlabeled / dropped → 0
REBALANCED_BACKGROUND_ID = 0


def _save_mask_png(mask: np.ndarray, out_path: Path) -> None:
    """Save a single-channel uint8 mask without deprecated Pillow kwargs.

    Args:
        mask: 2D ``uint8`` label map.
        out_path: Output ``.png`` path.

    Returns:
        None.

    Raises:
        OSError: If the file cannot be written.
    """
    arr = np.asarray(mask, dtype=np.uint8)
    Image.fromarray(arr).save(out_path)


def collect_class_titles(ann_dir: Path) -> list[str]:
    """Gather unique classTitle strings from all JSON files under ann_dir.

    Args:
        ann_dir: Directory containing ``*.jpg.json`` annotation files.

    Returns:
        Sorted list of unique class names (deterministic label order).

    Raises:
        FileNotFoundError: If ann_dir does not exist.
    """
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"Annotation directory not found: {ann_dir}")
    titles: set[str] = set()
    for jpath in sorted(ann_dir.glob("*.json")):
        with jpath.open(encoding="utf-8") as f:
            data = json.load(f)
        for obj in data.get("objects", []):
            t = obj.get("classTitle")
            if t is not None:
                titles.add(str(t))
    return sorted(titles)


def build_class_to_id(class_titles: list[str]) -> dict[str, int]:
    """Map each class name to integer id 0..len-1.

    Args:
        class_titles: Ordered list of class names (typically sorted).

    Returns:
        Mapping from class name to contiguous id starting at 0.

    Raises:
        ValueError: If the number of classes exceeds 103 (ingredient slots).
    """
    if len(class_titles) > 103:
        raise ValueError(
            f"Expected at most 103 ingredient classes, got {len(class_titles)}"
        )
    return {name: idx for idx, name in enumerate(class_titles)}


def decode_bitmap_mask(data_b64: str) -> np.ndarray:
    """Decode Supervisely bitmap.data to a binary uint8 mask.

    DatasetNinja / Supervisely exports often use base64(zlib(PNG)), not raw PNG.

    Args:
        data_b64: Base64-encoded payload (zlib-wrapped PNG or raw PNG).

    Returns:
        2D array of shape (H, W) with values 0 or 1.

    Raises:
        ValueError: If decoding fails or the image is empty.
    """
    try:
        raw = base64.b64decode(data_b64, validate=True)
    except Exception as exc:  # noqa: BLE001 — surface as clear error
        raise ValueError("Invalid base64 in bitmap.data") from exc
    try:
        im = Image.open(io.BytesIO(raw))
        im.load()
    except Exception:
        try:
            raw = zlib.decompress(raw)
        except zlib.error as exc:
            raise ValueError("bitmap.data is neither PNG nor zlib(PNG)") from exc
        im = Image.open(io.BytesIO(raw))
    if im.mode not in ("L", "1", "RGBA", "P"):
        im = im.convert("RGBA")
    gray = np.array(im.convert("L"), dtype=np.uint8)
    if gray.size == 0:
        raise ValueError("Decoded bitmap is empty")
    return (gray > 0).astype(np.uint8)


def resolve_canvas_size(
    ann_path: Path,
    ann_size: dict[str, Any],
    img_dir: Path | None,
) -> tuple[int, int]:
    """Choose (width, height) for the label canvas.

    Prefers the paired JPEG dimensions when the file exists (see dataset.md).

    Args:
        ann_path: Path to ``*.jpg.json`` (used to infer image basename).
        ann_size: JSON ``size`` dict with ``width`` and ``height``.
        img_dir: Optional ``img`` directory; if None, only JSON size is used.

    Returns:
        Tuple ``(width, height)``.

    Raises:
        KeyError: If ann_size lacks width/height.
    """
    w, h = int(ann_size["width"]), int(ann_size["height"])
    if img_dir is not None:
        stem = ann_path.name.removesuffix(".json")
        img_path = img_dir / stem
        if img_path.is_file():
            with Image.open(img_path) as im:
                iw, ih = im.size
            if (iw, ih) != (w, h):
                # Still trust the image file for training alignment.
                return iw, ih
    return w, h


def rasterize_annotation(
    data: dict[str, Any],
    class_to_id: dict[str, int],
    canvas_w: int,
    canvas_h: int,
) -> np.ndarray:
    """Paint all instance bitmaps onto a full-size semantic mask.

    Later objects in ``objects`` overwrite earlier pixels (overlap rule).

    Args:
        data: Loaded JSON root dict.
        class_to_id: classTitle → id in 0..102.
        canvas_w: Label canvas width.
        canvas_h: Label canvas height.

    Returns:
        uint8 array shape (H, W) with values in [0, 103].

    Raises:
        KeyError: If an object has unknown classTitle.
    """
    canvas = np.full((canvas_h, canvas_w), BACKGROUND_ID, dtype=np.uint8)
    for obj in data.get("objects", []):
        if obj.get("geometryType") != "bitmap":
            continue
        title = obj.get("classTitle")
        if title is None:
            continue
        if title not in class_to_id:
            raise KeyError(f"Unknown classTitle in annotation: {title!r}")
        cid = class_to_id[title]
        bmap = obj.get("bitmap") or {}
        origin = bmap.get("origin", [0, 0])
        ox, oy = int(origin[0]), int(origin[1])
        patch = decode_bitmap_mask(bmap["data"])
        ph, pw = patch.shape
        y1, x1 = oy, ox
        y2, x2 = oy + ph, ox + pw
        cy0, cy1 = max(0, y1), min(canvas_h, y2)
        cx0, cx1 = max(0, x1), min(canvas_w, x2)
        if cy0 >= cy1 or cx0 >= cx1:
            continue
        py0, px0 = cy0 - y1, cx0 - x1
        sl = patch[py0 : py0 + (cy1 - cy0), px0 : px0 + (cx1 - cx0)]
        region = canvas[cy0:cy1, cx0:cx1]
        region[sl > 0] = cid
    return canvas


def load_rebalanced_csv(
    csv_path: Path,
) -> tuple[dict[str, int | None], dict[int, str], list[str], dict[str, int]]:
    """Parse ``class_mapping.csv`` for the rebalanced FoodSeg103 taxonomy.

    Args:
        csv_path: CSV with columns ``source_class``, ``target_class``, ``kept``, ``new_class_id``.

    Returns:
        Tuple ``(source_to_train_id, train_id_to_target_name, dropped_sources, target_title_to_id)``.
        Dropped rows map to ``None`` in ``source_to_train_id``; kept rows map to ``1..76``.
        ``target_title_to_id`` maps merged **target** names (e.g. ``vegetable``, ``mushroom``)
        to train ids so JSON ``classTitle`` fields that already use group names resolve.

    Raises:
        FileNotFoundError: If ``csv_path`` is missing.
        ValueError: If the CSV is malformed.
    """
    if not csv_path.is_file():
        raise FileNotFoundError(f"Rebalance CSV not found: {csv_path}")
    source_to_id: dict[str, int | None] = {}
    id_to_target: dict[int, str] = {}
    target_title_to_id: dict[str, int] = {}
    dropped: list[str] = []
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"source_class", "target_class", "kept", "new_class_id"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"CSV must have columns {required}, got {reader.fieldnames!r}")
        for row in reader:
            src = (row.get("source_class") or "").strip()
            if not src:
                continue
            kept = str(row.get("kept", "")).strip().lower() in ("true", "1", "yes")
            if kept:
                raw_id = (row.get("new_class_id") or "").strip()
                if not raw_id:
                    raise ValueError(f"kept=True but empty new_class_id for {src!r}")
                tid = int(float(raw_id))
                source_to_id[src] = tid
                tgt = (row.get("target_class") or "").strip() or src
                id_to_target[tid] = tgt
                target_title_to_id[tgt] = tid
            else:
                source_to_id[src] = None
                dropped.append(src)
    return source_to_id, id_to_target, dropped, target_title_to_id


def validate_image_readable(img_path: Path) -> bool:
    """Return True if Pillow can open and load the image file.

    Args:
        img_path: Path to an image file.

    Returns:
        ``True`` if readable, else ``False``.
    """
    try:
        with Image.open(img_path) as im:
            im.verify()
        with Image.open(img_path) as im:
            im.load()
            if im.size[0] < 1 or im.size[1] < 1:
                return False
    except (OSError, ValueError, SyntaxError):
        return False
    return True


def remove_orphan_masks(split_dir: Path) -> int:
    """Delete ``mask/*.png`` with no matching ``ann/*.jpg.json``.

    Args:
        split_dir: Split root (contains optional ``mask/``).

    Returns:
        Number of mask files removed.
    """
    mask_dir = split_dir / "mask"
    ann_dir = split_dir / "ann"
    if not mask_dir.is_dir():
        return 0
    n = 0
    for mp in list(mask_dir.glob("*.png")):
        ann_path = ann_dir / f"{mp.stem}.jpg.json"
        if not ann_path.is_file():
            mp.unlink()
            n += 1
    return n


def clean_split_corrupt(split_dir: Path) -> list[str]:
    """Remove broken ``img``/``ann`` pairs and orphan files under one split.

    Deletes samples when: JPEG unreadable, JSON unreadable, missing paired file.
    Also removes orphan ``img`` without ``ann`` and stale ``mask`` files.

    Args:
        split_dir: e.g. ``.../foodseg103_rebalanced/train``.

    Returns:
        List of removed image basenames (e.g. ``00003985.jpg``) for logging.

    Raises:
        FileNotFoundError: If ``ann`` or ``img`` is missing entirely.
    """
    ann_dir = split_dir / "ann"
    img_dir = split_dir / "img"
    mask_dir = split_dir / "mask"
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"Missing ann dir: {ann_dir}")
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Missing img dir: {img_dir}")
    mask_dir.mkdir(parents=True, exist_ok=True)

    removed: list[str] = []

    def _delete_sample(stem_jpg: str) -> None:
        ip = img_dir / stem_jpg
        ap = ann_dir / f"{stem_jpg}.json"
        mp = mask_dir / (Path(stem_jpg).stem + ".png")
        for p in (ap, ip, mp):
            if p.is_file():
                p.unlink()
        if stem_jpg not in removed:
            removed.append(stem_jpg)

    for ann_path in list(ann_dir.glob("*.json")):
        stem_jpg = ann_path.name.removesuffix(".json")
        img_path = img_dir / stem_jpg
        try:
            with ann_path.open(encoding="utf-8") as f:
                json.load(f)
        except (json.JSONDecodeError, OSError):
            _delete_sample(stem_jpg)
            continue
        if not img_path.is_file():
            ann_path.unlink()
            mp = mask_dir / (Path(stem_jpg).stem + ".png")
            if mp.is_file():
                mp.unlink()
            removed.append(stem_jpg)
            continue
        if not validate_image_readable(img_path):
            _delete_sample(stem_jpg)

    for img_path in list(img_dir.glob("*.jpg")):
        ann_path = ann_dir / f"{img_path.name}.json"
        if not ann_path.is_file():
            img_path.unlink()
            mp = mask_dir / (img_path.stem + ".png")
            if mp.is_file():
                mp.unlink()
            if img_path.name not in removed:
                removed.append(img_path.name)

    remove_orphan_masks(split_dir)
    return removed


def rasterize_rebalanced_annotation(
    data: dict[str, Any],
    source_to_train_id: dict[str, int | None],
    target_title_to_id: dict[str, int],
    canvas_w: int,
    canvas_h: int,
) -> np.ndarray:
    """Rasterize JSON to a mask with rebalanced ids (1..76) and background 0.

    Skips objects whose ``classTitle`` is dropped (mapped to ``None``).
    Later objects overwrite earlier pixels.

    Args:
        data: Loaded Supervisely-style JSON root.
        source_to_train_id: Original ingredient ``classTitle`` → train id or ``None``.
        target_title_to_id: Merged group name (``target_class`` from CSV) → train id.
        canvas_w: Canvas width.
        canvas_h: Canvas height.

    Returns:
        ``uint8`` array of shape ``(H, W)`` with values in ``{0, 1..76}``.

    Raises:
        KeyError: If an object references an unknown ``classTitle``.
    """
    canvas = np.full((canvas_h, canvas_w), REBALANCED_BACKGROUND_ID, dtype=np.uint8)
    for obj in data.get("objects", []):
        if obj.get("geometryType") != "bitmap":
            continue
        title = obj.get("classTitle")
        if title is None:
            continue
        title = str(title)
        if title in source_to_train_id:
            cid = source_to_train_id[title]
            if cid is None:
                continue
        elif title in target_title_to_id:
            cid = target_title_to_id[title]
        else:
            raise KeyError(f"Unknown classTitle in annotation: {title!r}")
        bmap = obj.get("bitmap") or {}
        origin = bmap.get("origin", [0, 0])
        ox, oy = int(origin[0]), int(origin[1])
        patch = decode_bitmap_mask(bmap["data"])
        ph, pw = patch.shape
        y1, x1 = oy, ox
        y2, x2 = oy + ph, ox + pw
        cy0, cy1 = max(0, y1), min(canvas_h, y2)
        cx0, cx1 = max(0, x1), min(canvas_w, x2)
        if cy0 >= cy1 or cx0 >= cx1:
            continue
        py0, px0 = cy0 - y1, cx0 - x1
        sl = patch[py0 : py0 + (cy1 - cy0), px0 : px0 + (cx1 - cx0)]
        region = canvas[cy0:cy1, cx0:cx1]
        region[sl > 0] = cid
    return canvas


def convert_split_rebalanced(
    split_dir: Path,
    source_to_train_id: dict[str, int | None],
    target_title_to_id: dict[str, int],
    overwrite: bool,
    delete_on_error: bool,
) -> tuple[int, list[str]]:
    """Write rebalanced PNG masks for every ``ann/*.jpg.json`` with a valid JPEG.

    Args:
        split_dir: Split directory with ``ann/``, ``img/``.
        source_to_train_id: Mapping including dropped classes (``None``).
        target_title_to_id: Group ``classTitle`` aliases → train id.
        overwrite: Regenerate existing masks.
        delete_on_error: If rasterize/save fails, delete img/ann/mask for that stem.

    Returns:
        Tuple ``(masks_written, removed_on_error)``.
    """
    ann_dir = split_dir / "ann"
    img_dir = split_dir / "img"
    mask_dir = split_dir / "mask"
    mask_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    err_removed: list[str] = []
    for ann_path in sorted(ann_dir.glob("*.json")):
        stem_jpg = ann_path.name.removesuffix(".json")
        img_path = img_dir / stem_jpg
        out_name = Path(stem_jpg).stem + ".png"
        out_path = mask_dir / out_name
        if out_path.exists() and not overwrite:
            continue
        if not img_path.is_file() or not validate_image_readable(img_path):
            continue
        try:
            with ann_path.open(encoding="utf-8") as f:
                data = json.load(f)
            size = data.get("size") or {}
            cw, ch = resolve_canvas_size(
                ann_path,
                size,
                img_dir if img_dir.is_dir() else None,
            )
            mask = rasterize_rebalanced_annotation(
                data, source_to_train_id, target_title_to_id, cw, ch
            )
            _save_mask_png(mask, out_path)
            written += 1
        except (KeyError, ValueError, OSError, json.JSONDecodeError) as exc:
            print(
                f"[rebalanced] skip/delete {stem_jpg}: {exc}",
                file=sys.stderr,
            )
            if delete_on_error:
                for p in (ann_path, img_path, out_path):
                    if p.is_file():
                        p.unlink()
                err_removed.append(stem_jpg)
    return written, err_removed


def _stable_color_for_title(title: str) -> str:
    """Pick a deterministic ``#RRGGBB`` color from the title string.

    Args:
        title: Class display name.

    Returns:
        Hex color string.
    """
    h = hashlib.sha256(title.encode("utf-8")).hexdigest()
    return f"#{h[:6]}"


def update_rebalanced_meta(data_root: Path, train_id_to_target: dict[int, str]) -> None:
    """Replace ``meta.json`` ``classes`` with the 76 rebalanced training classes.

    Preserves ``tags``, ``projectType``, and ``projectSettings``.

    Args:
        data_root: Dataset root containing ``meta.json``.
        train_id_to_target: Train ids ``1..76`` → canonical target name.

    Returns:
        None.

    Raises:
        FileNotFoundError: If ``meta.json`` is missing.
    """
    meta_path = data_root / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta.json not found: {meta_path}")
    with meta_path.open(encoding="utf-8") as f:
        meta = json.load(f)
    old_classes = meta.get("classes") or []
    old_by_title = {str(c.get("title")): c for c in old_classes if c.get("title")}

    new_classes: list[dict[str, Any]] = []
    for tid in sorted(train_id_to_target.keys()):
        title = train_id_to_target[tid]
        oc = old_by_title.get(title)
        color = (
            str(oc.get("color"))
            if isinstance(oc, dict) and oc.get("color")
            else _stable_color_for_title(title)
        )
        new_classes.append(
            {
                "title": title,
                "shape": "bitmap",
                "color": color,
                "geometry_config": {},
                "id": int(tid),
                "hotkey": "",
            }
        )
    meta["classes"] = new_classes
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def write_rebalanced_class_mapping_json(
    data_root: Path,
    source_to_train_id: dict[str, int | None],
    train_id_to_target: dict[int, str],
    dropped_sources: list[str],
    target_title_to_id: dict[str, int],
) -> None:
    """Write ``class_mapping.json`` for the rebalanced dataset.

    Args:
        data_root: Dataset root (output file path).
        source_to_train_id: Includes ``None`` for dropped sources.
        train_id_to_target: Train id → display name.
        dropped_sources: Source class names removed from labels.
        target_title_to_id: Merged labels usable as JSON ``classTitle``.

    Returns:
        None.
    """
    kept_map = {k: v for k, v in source_to_train_id.items() if v is not None}
    mapping_path = data_root / "class_mapping.json"
    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "schema": "foodseg103_rebalanced",
                "class_to_id": kept_map,
                "target_title_to_id": dict(sorted(target_title_to_id.items())),
                "id_to_class": {str(i): train_id_to_target[i] for i in sorted(train_id_to_target)},
                "background_id": REBALANCED_BACKGROUND_ID,
                "num_foreground_classes": len(train_id_to_target),
                "num_classes": len(train_id_to_target) + 1,
                "dropped_source_classes": sorted(set(dropped_sources)),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


def convert_split(
    split_dir: Path,
    class_to_id: dict[str, int],
    overwrite: bool,
) -> int:
    """Write PNG masks for every annotation in ``split_dir/ann``.

    Args:
        split_dir: e.g. ``.../foodseg103/train`` (must contain ``ann/``).
        class_to_id: classTitle → id.
        overwrite: If False, skip when output PNG already exists.

    Returns:
        Number of mask files written.

    Raises:
        FileNotFoundError: If ``ann`` is missing.
    """
    ann_dir = split_dir / "ann"
    img_dir = split_dir / "img"
    mask_dir = split_dir / "mask"
    mask_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for ann_path in sorted(ann_dir.glob("*.json")):
        out_name = ann_path.name.removesuffix(".json").removesuffix(".jpg") + ".png"
        out_path = mask_dir / out_name
        if out_path.exists() and not overwrite:
            continue
        with ann_path.open(encoding="utf-8") as f:
            data = json.load(f)
        size = data.get("size") or {}
        cw, ch = resolve_canvas_size(
            ann_path,
            size,
            img_dir if img_dir.is_dir() else None,
        )
        mask = rasterize_annotation(data, class_to_id, cw, ch)
        _save_mask_png(mask, out_path)
        written += 1
    return written


def main() -> None:
    """CLI: build class map from annotations and export PNG masks."""
    root = Path(__file__).resolve().parent
    default_root = root.parent / "dataset" / "foodseg103-full"
    default_reb = root.parent / "dataset" / "foodseg103_rebalanced"

    parser = argparse.ArgumentParser(
        description=(
            "FoodSeg103: JSON ann to grayscale PNG masks "
            "(standard: ids 0-102 + 103=bg; --rebalanced: ids 1-76 + 0=bg)."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Path to dataset folder (contains train/, test/). "
        "Default: foodseg103-full, or foodseg103_rebalanced with --rebalanced.",
    )
    parser.add_argument(
        "--rebalanced",
        action="store_true",
        help="Use class_mapping.csv remap (76 fg + bg 0); refresh meta.json + class_mapping.json.",
    )
    parser.add_argument(
        "--rebalance-csv",
        type=Path,
        default=None,
        help="Path to class_mapping.csv (default: <data-root>/class_mapping.csv).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove corrupt/mismatched img+ann (+stale mask) before converting.",
    )
    parser.add_argument(
        "--delete-on-mask-error",
        action="store_true",
        help="With --rebalanced: delete sample if mask rasterization fails.",
    )
    parser.add_argument(
        "--no-update-meta",
        action="store_true",
        help="With --rebalanced: skip rewriting meta.json classes list.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "test", "both"),
        default="train",
        help="Which split to convert.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate masks even if PNG already exists.",
    )
    args = parser.parse_args()

    data_root = args.data_root
    if data_root is None:
        data_root = default_reb if args.rebalanced else default_root
    data_root = data_root.resolve()
    if not data_root.is_dir():
        print(f"data-root not found: {data_root}", file=sys.stderr)
        sys.exit(1)

    splits = ["train", "test"] if args.split == "both" else [args.split]
    ann_dirs = [data_root / s / "ann" for s in splits]
    for d in ann_dirs:
        if not d.is_dir():
            print(f"Missing: {d}", file=sys.stderr)
            sys.exit(1)

    if args.rebalanced:
        csv_path = (
            args.rebalance_csv.resolve()
            if args.rebalance_csv is not None
            else (data_root / "class_mapping.csv")
        )
        source_to_id, id_to_target, dropped, target_to_id = load_rebalanced_csv(csv_path)
        if args.clean:
            for s in splits:
                rem = clean_split_corrupt(data_root / s)
                if rem:
                    print(f"{s}: clean removed {len(rem)} sample(s), e.g. {rem[:5]!r}")

        total = 0
        all_err: list[str] = []
        for s in splits:
            n, er = convert_split_rebalanced(
                data_root / s,
                source_to_id,
                target_to_id,
                args.overwrite,
                args.delete_on_mask_error,
            )
            print(f"{s}: wrote {n} masks -> {data_root / s / 'mask'}")
            total += n
            all_err.extend(er)
        write_rebalanced_class_mapping_json(
            data_root, source_to_id, id_to_target, dropped, target_to_id
        )
        if not args.no_update_meta:
            update_rebalanced_meta(data_root, id_to_target)
            print(f"meta.json classes updated -> {data_root / 'meta.json'}")
        else:
            print("meta.json left unchanged (--no-update-meta)")
        print(f"class_mapping.json -> {data_root / 'class_mapping.json'}")
        print(f"total masks written: {total}")
        if all_err:
            print(f"removed on raster error: {len(all_err)}", file=sys.stderr)
        return

    # --- standard 103-class FoodSeg103 ---
    all_titles: set[str] = set()
    for d in ann_dirs:
        all_titles.update(collect_class_titles(d))
    class_titles = sorted(all_titles)
    class_to_id = build_class_to_id(class_titles)

    mapping_path = data_root / "class_mapping.json"
    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "class_to_id": class_to_id,
                "id_to_class": {str(v): k for k, v in class_to_id.items()},
                "background_id": BACKGROUND_ID,
                "num_ingredient_classes": len(class_titles),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    total = 0
    for s in splits:
        n = convert_split(data_root / s, class_to_id, args.overwrite)
        print(f"{s}: wrote {n} masks -> {data_root / s / 'mask'}")
        total += n
    print(f"class_mapping.json -> {mapping_path} ({len(class_titles)} classes)")
    print(f"total masks written: {total}")


if __name__ == "__main__":
    main()
