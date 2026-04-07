import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "source" / "dataset" / "foodseg103_rebalanced"
DOCS_DIR = DATA_DIR / "docs"
PATCH_DIR = DOCS_DIR / "patches"
QUALITY_DIR = PATCH_DIR / "quality_samples"

TEXTURE_GROUP_CLASSES = {
    "cabbage",
    "lettuce",
    "rape",
    "vegetable",
    "mushroom",
    "pork",
    "steak",
    "rice",
    "potato",
}


def ensure_dirs() -> None:
    """Create all required output directories.

    Args:
        None.
    Returns:
        None.
    Raises:
        OSError: If any directory cannot be created.
    """
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PATCH_DIR.mkdir(parents=True, exist_ok=True)
    QUALITY_DIR.mkdir(parents=True, exist_ok=True)


def load_mappings() -> tuple[dict, pd.DataFrame]:
    """Load class mapping json and csv files.

    Args:
        None.
    Returns:
        tuple[dict, pd.DataFrame]: Parsed JSON mapping and CSV mapping table.
    Raises:
        FileNotFoundError: If mapping files are missing.
        json.JSONDecodeError: If class_mapping.json is invalid.
    """
    with (DATA_DIR / "class_mapping.json").open("r", encoding="utf-8") as f:
        mapping_json = json.load(f)
    mapping_csv = pd.read_csv(DATA_DIR / "class_mapping.csv")
    return mapping_json, mapping_csv


def list_split_files(split: str) -> tuple[list[Path], list[Path], Path]:
    """List annotation and mask files for one split.

    Args:
        split: Split name, expected "train" or "test".
    Returns:
        tuple[list[Path], list[Path], Path]: Annotation files, mask files, image directory.
    Raises:
        FileNotFoundError: If split folders do not exist.
    """
    split_dir = DATA_DIR / split
    ann_dir = split_dir / "ann"
    mask_dir = split_dir / "mask"
    img_dir = split_dir / "img"
    if not split_dir.exists():
        raise FileNotFoundError(f"Split not found: {split_dir}")
    return sorted(ann_dir.glob("*.json")), sorted(mask_dir.glob("*.png")), img_dir


def extract_labels_from_annotation(ann_obj: dict) -> set[str]:
    """Extract class labels from one annotation object.

    Args:
        ann_obj: Annotation dictionary loaded from JSON file.
    Returns:
        set[str]: Class titles found in the annotation payload.
    Raises:
        TypeError: If ann_obj is not a dictionary.
    """
    if not isinstance(ann_obj, dict):
        raise TypeError("Annotation payload must be a dict.")

    labels: set[str] = set()
    for obj in ann_obj.get("objects", []):
        if isinstance(obj, dict):
            label = obj.get("classTitle") or obj.get("label")
            if isinstance(label, str) and label.strip():
                labels.add(label.strip())
    for shape in ann_obj.get("shapes", []):
        if isinstance(shape, dict):
            label = shape.get("label") or shape.get("classTitle")
            if isinstance(label, str) and label.strip():
                labels.add(label.strip())
    return labels


def scan_json_presence(ann_files: list[Path], class_to_id: dict[str, int]) -> dict[int, int]:
    """Count per-class annotation presence in JSON files.

    Args:
        ann_files: List of annotation JSON files.
        class_to_id: Mapping from class name to target class id.
    Returns:
        dict[int, int]: Number of files where each class appears.
    Raises:
        OSError: If file reading fails unexpectedly.
    """
    presence = defaultdict(int)
    for ann_file in ann_files:
        with ann_file.open("r", encoding="utf-8") as f:
            ann_obj = json.load(f)
        labels = extract_labels_from_annotation(ann_obj)
        unique_ids = {class_to_id[name] for name in labels if name in class_to_id}
        for cid in unique_ids:
            presence[cid] += 1
    return dict(presence)


def scan_mask_stats(mask_files: list[Path]) -> tuple[dict[int, int], dict[int, int]]:
    """Count per-class presence and total pixels from mask files.

    Args:
        mask_files: List of grayscale mask PNG files.
    Returns:
        tuple[dict[int, int], dict[int, int]]: Presence-by-file and pixel totals by class id.
    Raises:
        RuntimeError: If a mask cannot be decoded.
    """
    presence = defaultdict(int)
    pixels = defaultdict(int)
    for mask_path in mask_files:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Cannot read mask: {mask_path}")
        unique_ids, counts = np.unique(mask, return_counts=True)
        for cid, cnt in zip(unique_ids, counts):
            cid_i = int(cid)
            if cid_i == 0:
                continue
            presence[cid_i] += 1
            pixels[cid_i] += int(cnt)
    return dict(presence), dict(pixels)


def resolve_image_path(img_dir: Path, stem: str) -> Path | None:
    """Resolve image path from a file stem.

    Args:
        img_dir: Directory containing raw images.
        stem: File stem from the corresponding mask.
    Returns:
        Path | None: Existing image path or None if not found.
    Raises:
        OSError: If filesystem access fails unexpectedly.
    """
    candidates = [img_dir / f"{stem}.jpg", img_dir / f"{stem}.png", img_dir / f"{stem}.jpeg"]
    for path in candidates:
        if path.exists():
            return path
    return None


def make_gallery(images: list[np.ndarray], output_path: Path, tile_size: int = 160, cols: int = 6) -> None:
    """Compose a simple image gallery into one PNG.

    Args:
        images: List of BGR patches.
        output_path: Path to save gallery PNG.
        tile_size: Target square size for each tile.
        cols: Number of columns in the gallery.
    Returns:
        None.
    Raises:
        ValueError: If tile_size or cols is invalid.
    """
    if tile_size <= 0 or cols <= 0:
        raise ValueError("tile_size and cols must be positive.")
    if not images:
        canvas = np.full((tile_size, tile_size, 3), 245, dtype=np.uint8)
        cv2.putText(canvas, "No samples", (10, tile_size // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2)
        cv2.imwrite(str(output_path), canvas)
        return
    rows = int(np.ceil(len(images) / cols))
    gallery = np.full((rows * tile_size, cols * tile_size, 3), 240, dtype=np.uint8)
    for idx, img in enumerate(images):
        resized = cv2.resize(img, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
        r = idx // cols
        c = idx % cols
        gallery[r * tile_size : (r + 1) * tile_size, c * tile_size : (c + 1) * tile_size] = resized
    cv2.imwrite(str(output_path), gallery)


def build_dead_class_table(
    id_to_class: dict[str, str],
    mapping_csv: pd.DataFrame,
    json_presence: dict[int, int],
    mask_presence: dict[int, int],
    mask_pixels: dict[int, int],
) -> pd.DataFrame:
    """Build class-by-class dead analysis table.

    Args:
        id_to_class: Mapping from class id string to class name.
        mapping_csv: Raw class mapping CSV table.
        json_presence: Class presence counts from JSON annotations.
        mask_presence: Class presence counts from masks.
        mask_pixels: Class pixel totals from masks.
    Returns:
        pd.DataFrame: Table with required columns and action suggestion.
    Raises:
        KeyError: If expected columns are missing from mapping CSV.
    """
    if "target_class" not in mapping_csv.columns:
        raise KeyError("class_mapping.csv must contain column 'target_class'.")

    mapped_target_names = set(mapping_csv["target_class"].dropna().astype(str).tolist())
    rows = []
    for cid_str, cname in sorted(id_to_class.items(), key=lambda x: int(x[0])):
        cid = int(cid_str)
        in_mapping = cname in mapped_target_names
        ann_count = int(json_presence.get(cid, 0))
        mask_count = int(mask_presence.get(cid, 0))
        px = int(mask_pixels.get(cid, 0))
        if ann_count == 0 and mask_count == 0:
            action = "drop_from_train_ontology"
        elif ann_count > 0 and mask_count == 0:
            action = "investigate_remap_or_mask_generation"
        else:
            action = "keep"
        rows.append(
            {
                "class_name": cname,
                "class_id": cid,
                "in_mapping": int(in_mapping),
                "in_json_ann": ann_count,
                "in_mask_pixels": px,
                "presence_count": mask_count,
                "action": action,
            }
        )
    return pd.DataFrame(rows)


def build_split_coverage(
    id_to_class: dict[str, str],
    train_presence: dict[int, int],
    test_presence: dict[int, int],
    train_pixels: dict[int, int],
    test_pixels: dict[int, int],
) -> pd.DataFrame:
    """Build train/val class coverage table with flags.

    Args:
        id_to_class: Mapping from class id string to class name.
        train_presence: Train presence count by class id.
        test_presence: Validation presence count by class id.
        train_pixels: Train pixel totals by class id.
        test_pixels: Validation pixel totals by class id.
    Returns:
        pd.DataFrame: Coverage table with required metrics and flags.
    Raises:
        ZeroDivisionError: Never raised because division is guarded.
    """
    rows = []
    for cid_str, cname in sorted(id_to_class.items(), key=lambda x: int(x[0])):
        cid = int(cid_str)
        train_p = int(train_presence.get(cid, 0))
        val_p = int(test_presence.get(cid, 0))
        train_px = int(train_pixels.get(cid, 0))
        val_px = int(test_pixels.get(cid, 0))
        train_mean = float(train_px / train_p) if train_p > 0 else 0.0
        val_mean = float(val_px / val_p) if val_p > 0 else 0.0

        flags = []
        if train_p == 0 and val_p == 0:
            flags.append("dead")
        if train_p == 0 and val_p > 0:
            flags.append("missing_in_train")
        if train_p > 0 and val_p == 0:
            flags.append("missing_in_val")
        if train_p <= 2 and train_p > 0:
            flags.append("ultra_rare")
        elif train_p <= 5 and train_p > 0:
            flags.append("rare")
        if 0 < train_mean < 1000:
            flags.append("pixel_starved")

        rows.append(
            {
                "class_id": cid,
                "class_name": cname,
                "train_presence_count": train_p,
                "val_presence_count": val_p,
                "train_total_pixels": train_px,
                "val_total_pixels": val_px,
                "mean_pixels_when_present_train": round(train_mean, 4),
                "mean_pixels_when_present_val": round(val_mean, 4),
                "flags": ",".join(flags),
            }
        )
    return pd.DataFrame(rows)


def run_patch_quality_scan(
    train_masks: list[Path],
    id_to_class: dict[str, str],
    img_dir: Path,
) -> pd.DataFrame:
    """Scan texture patches and compute quality metrics.

    Args:
        train_masks: Training mask files.
        id_to_class: Mapping from class id string to class name.
        img_dir: Directory containing train images.
    Returns:
        pd.DataFrame: Patch quality table with required fields.
    Raises:
        RuntimeError: If a required image cannot be decoded.
    """
    class_lookup = {int(k): v for k, v in id_to_class.items()}
    target_ids = {cid for cid, cname in class_lookup.items() if cname in TEXTURE_GROUP_CLASSES}

    patch_rows = []
    blurry_samples = []
    tight_samples = []
    low_fg_samples = []

    for mask_path in train_masks:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        present_ids = set(int(x) for x in np.unique(mask))
        target_present = sorted([cid for cid in present_ids if cid in target_ids])
        if not target_present:
            continue

        img_path = resolve_image_path(img_dir, mask_path.stem)
        if img_path is None:
            continue
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Cannot read image: {img_path}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        total_fg = int(np.count_nonzero(mask > 0))
        if total_fg == 0:
            continue

        for cid in target_present:
            class_mask = (mask == cid).astype(np.uint8)
            n_labels, _, stats, _ = cv2.connectedComponentsWithStats(class_mask, connectivity=8)
            for comp_idx in range(1, n_labels):
                x, y, w, h, area = stats[comp_idx].tolist()
                if area < 300:
                    continue
                patch_img = image[y : y + h, x : x + w]
                patch_gray = gray[y : y + h, x : x + w]
                if patch_img.size == 0:
                    continue

                blur_score = float(cv2.Laplacian(patch_gray, cv2.CV_64F).var())
                bbox_fill_ratio = float(area / max(1, w * h))
                fg_ratio = float(area / total_fg)

                flags = []
                if blur_score < 80.0:
                    flags.append("too_blurry")
                if bbox_fill_ratio > 0.93:
                    flags.append("too_tight")
                if fg_ratio < 0.05:
                    flags.append("low_fg")
                quality_flag = "|".join(flags) if flags else "ok"
                patch_id = f"{mask_path.stem}_{cid}_{comp_idx}"

                patch_rows.append(
                    {
                        "patch_id": patch_id,
                        "class_name": class_lookup[cid],
                        "fg_ratio": round(fg_ratio, 6),
                        "blur_score": round(blur_score, 6),
                        "bbox_fill_ratio": round(bbox_fill_ratio, 6),
                        "quality_flag": quality_flag,
                    }
                )

                if "too_blurry" in flags and len(blurry_samples) < 36:
                    blurry_samples.append(patch_img)
                if "too_tight" in flags and len(tight_samples) < 36:
                    tight_samples.append(patch_img)
                if "low_fg" in flags and len(low_fg_samples) < 36:
                    low_fg_samples.append(patch_img)

    make_gallery(blurry_samples, PATCH_DIR / "blurry_patches_gallery.png")
    make_gallery(tight_samples, PATCH_DIR / "too_tight_patches_gallery.png")
    make_gallery(low_fg_samples, PATCH_DIR / "low_foreground_patches_gallery.png")
    return pd.DataFrame(patch_rows)


def build_issue_summary(
    dead_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    patch_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build final issue summary table for reporting.

    Args:
        dead_df: Dead-class audit table.
        coverage_df: Split class coverage table.
        patch_df: Texture patch quality table.
    Returns:
        pd.DataFrame: Summary with issue_type, item, evidence, action.
    Raises:
        ValueError: If required columns are missing.
    """
    issues = []
    required = {"class_name", "action", "in_json_ann", "in_mask_pixels", "presence_count"}
    if not required.issubset(set(dead_df.columns)):
        raise ValueError("dead_df missing required columns.")

    dead_rows = dead_df[dead_df["action"] == "drop_from_train_ontology"]
    for _, row in dead_rows.iterrows():
        evidence = f"json={int(row['in_json_ann'])}, pixels={int(row['in_mask_pixels'])}, presence={int(row['presence_count'])}"
        issues.append(
            {
                "issue_type": "dead_class",
                "item": row["class_name"],
                "evidence": evidence,
                "action": "drop_from_train_ontology",
            }
        )

    missing_train = coverage_df[coverage_df["flags"].str.contains("missing_in_train", na=False)]
    for _, row in missing_train.iterrows():
        evidence = f"train_presence={int(row['train_presence_count'])}, val_presence={int(row['val_presence_count'])}"
        issues.append(
            {
                "issue_type": "split_issue",
                "item": row["class_name"],
                "evidence": evidence,
                "action": "move_or_resplit_samples_to_train",
            }
        )

    missing_val = coverage_df[coverage_df["flags"].str.contains("missing_in_val", na=False)]
    for _, row in missing_val.iterrows():
        evidence = f"train_presence={int(row['train_presence_count'])}, val_presence={int(row['val_presence_count'])}"
        issues.append(
            {
                "issue_type": "split_issue",
                "item": row["class_name"],
                "evidence": evidence,
                "action": "add_validation_samples",
            }
        )

    rare = coverage_df[coverage_df["flags"].str.contains("ultra_rare|rare", na=False)]
    for _, row in rare.iterrows():
        evidence = f"train_presence={int(row['train_presence_count'])}, train_pixels={int(row['train_total_pixels'])}"
        issues.append(
            {
                "issue_type": "class_imbalance",
                "item": row["class_name"],
                "evidence": evidence,
                "action": "augment_or_merge_class",
            }
        )

    for flag in ["too_blurry", "too_tight", "low_fg"]:
        c = int(patch_df["quality_flag"].str.contains(flag, na=False).sum()) if not patch_df.empty else 0
        issues.append(
            {
                "issue_type": "patch_quality",
                "item": flag,
                "evidence": f"patch_count={c}",
                "action": "update_patch_extraction_policy",
            }
        )
    return pd.DataFrame(issues)


def main() -> None:
    """Run full dataset audit and write outputs into docs folder.

    Args:
        None.
    Returns:
        None.
    Raises:
        RuntimeError: If critical dataset files are unreadable.
    """
    ensure_dirs()
    mapping_json, mapping_csv = load_mappings()
    id_to_class = mapping_json["id_to_class"]
    class_to_id = mapping_json["target_title_to_id"]

    train_anns, train_masks, train_img_dir = list_split_files("train")
    test_anns, test_masks, _ = list_split_files("test")

    print(f"train ann={len(train_anns)}, train masks={len(train_masks)}")
    print(f"test ann={len(test_anns)}, test masks={len(test_masks)}")

    train_json_presence = scan_json_presence(train_anns, class_to_id)
    test_json_presence = scan_json_presence(test_anns, class_to_id)
    all_json_presence = defaultdict(int)
    for cid, c in train_json_presence.items():
        all_json_presence[cid] += c
    for cid, c in test_json_presence.items():
        all_json_presence[cid] += c

    train_presence, train_pixels = scan_mask_stats(train_masks)
    test_presence, test_pixels = scan_mask_stats(test_masks)
    all_mask_presence = defaultdict(int)
    all_mask_pixels = defaultdict(int)
    for cid, c in train_presence.items():
        all_mask_presence[cid] += c
    for cid, c in test_presence.items():
        all_mask_presence[cid] += c
    for cid, c in train_pixels.items():
        all_mask_pixels[cid] += c
    for cid, c in test_pixels.items():
        all_mask_pixels[cid] += c

    dead_df = build_dead_class_table(
        id_to_class=id_to_class,
        mapping_csv=mapping_csv,
        json_presence=dict(all_json_presence),
        mask_presence=dict(all_mask_presence),
        mask_pixels=dict(all_mask_pixels),
    )
    dead_df.to_csv(DOCS_DIR / "class_dead_audit.csv", index=False)
    dead_df[dead_df["class_name"] == "seaweed"].to_csv(DOCS_DIR / "seaweed_check.csv", index=False)

    coverage_df = build_split_coverage(
        id_to_class=id_to_class,
        train_presence=train_presence,
        test_presence=test_presence,
        train_pixels=train_pixels,
        test_pixels=test_pixels,
    )
    coverage_df.to_csv(DOCS_DIR / "split_class_coverage.csv", index=False)
    coverage_df[coverage_df["flags"].str.contains("missing_in_train", na=False)].to_csv(
        DOCS_DIR / "classes_missing_in_train.csv", index=False
    )
    coverage_df[coverage_df["flags"].str.contains("ultra_rare|rare", na=False)].to_csv(
        DOCS_DIR / "rare_classes_train.csv", index=False
    )

    patch_df = run_patch_quality_scan(train_masks=train_masks, id_to_class=id_to_class, img_dir=train_img_dir)
    patch_df.to_csv(DOCS_DIR / "texture_patches_quality.csv", index=False)

    issue_df = build_issue_summary(dead_df=dead_df, coverage_df=coverage_df, patch_df=patch_df)
    issue_df.to_csv(DOCS_DIR / "issue_summary.csv", index=False)
    print("Audit completed. Outputs saved to docs/ and docs/patches/")


if __name__ == "__main__":
    main()
