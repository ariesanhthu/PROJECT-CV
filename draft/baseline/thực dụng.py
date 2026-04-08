# %%
#!pip -q install kagglehub albumentations opencv-python-headless pandas tqdm

# %% [markdown]
# Nhớ đổi DRIVE_PROJECT_DIR

# %%
# =========================
# CONFIG - CHANGED
# =========================
CONFIG = {
    # Kaggle dataset
    "KAGGLE_DATASET": "arischii05/cleaned-foodseg103",
    "DATASET_FOLDER_NAME": "foodseg103_rebalanced",
    # Checkpoint/log lưu trong Colab runtime
    # Chỉ giữ được trong cùng runtime, không giữ qua factory reset / runtime mới
    "PROJECT_DIR": "/content/drive/MyDrive/tmp",
    "CKPT_DIRNAME": "checkpoints",
    "LOG_CSV_NAME": "train_log.csv",
    # Split cố định
    # Split vẫn nên giữ ở Drive hoặc file json đã tạo từ trước.
    # Nếu bạn đã có split_json_path từ cell trước thì giữ nguyên logic đó.
    "DRIVE_PROJECT_DIR": "/content/drive/MyDrive/tmp",
    "SPLIT_JSON_NAME": "splits_seed42.json",
    # Seed
    "SEED": 42,
    "VAL_RATIO": 0.10,
    # Dataset
    "NUM_CLASSES": 77,  # 0..76
    "BACKGROUND_ID": 0,
    # Memory-safe for Colab T4
    "IMAGE_SIZE": 384,  # 384 an toàn hơn 512, có thể tăng lại sau
    "BATCH_SIZE": 4,  # nếu OOM -> 2
    "NUM_WORKERS": 2,  # nếu RAM cao -> 0
    "PIN_MEMORY": True,
    # Model / transfer learning
    "MODEL_NAME": "BiSeNetV1_ResNet18_Pretrained",
    "PRETRAINED_BACKBONE": True,
    "USE_AUX_HEAD": True,
    "AUX_WEIGHT": 0.3,
    # ---- Training ----
    "EPOCHS": 80,
    "LR": 0.005,
    "MOMENTUM": 0.9,
    "WEIGHT_DECAY": 5e-4,
    "POLY_POWER": 0.9,
    "USE_AMP": True,
    # ------------------
    # Checkpoint
    "RESUME": True,
    "SAVE_BEST": True,
    "SAVE_LATEST_EVERY_EPOCH": True,
    # Memory optimization
    "GRAD_CLIP_NORM": 1.0,
}

# %%

# =========================
# IMPORTS
# =========================
import os
import gc
import json
import time
import math
import random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm.auto import tqdm

import kagglehub
from google.colab import drive

# =========================
# SEED
# =========================
random.seed(CONFIG["SEED"])
np.random.seed(CONFIG["SEED"])
torch.manual_seed(CONFIG["SEED"])
torch.cuda.manual_seed_all(CONFIG["SEED"])
torch.backends.cudnn.benchmark = True

# =========================
# MOUNT DRIVE
# =========================
drive.mount("/content/drive", force_remount=True)

# =========================
# PATHS
# =========================
PROJECT_DIR = Path(CONFIG["DRIVE_PROJECT_DIR"])
CKPT_DIR = PROJECT_DIR / CONFIG["CKPT_DIRNAME"]
PROJECT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# DOWNLOAD DATASET FROM KAGGLE
# =========================
kaggle_root = Path(kagglehub.dataset_download(CONFIG["KAGGLE_DATASET"]))
print("Kaggle root:", kaggle_root)

# thử tìm DATA_ROOT
candidate_1 = kaggle_root / CONFIG["DATASET_FOLDER_NAME"]
candidate_2 = kaggle_root

if candidate_1.exists():
    DATA_ROOT = candidate_1
elif candidate_2.exists():
    DATA_ROOT = candidate_2
else:
    raise FileNotFoundError("Không tìm thấy DATA_ROOT")

print("DATA_ROOT:", DATA_ROOT)

# in sơ bộ cây thư mục
print("\nTop-level files/folders:")
for p in sorted(DATA_ROOT.iterdir()):
    print(" -", p.name)

# %% [markdown]
# # Kiểm tra dataset rebalanced và load mapping

# %%
# =========================
# VERIFY DATASET
# =========================
required_paths = [
    DATA_ROOT / "class_mapping.json",
    DATA_ROOT / "train" / "img",
    DATA_ROOT / "train" / "mask",
    DATA_ROOT / "test" / "img",
    DATA_ROOT / "test" / "mask",
]

for p in required_paths:
    assert p.exists(), f"Thiếu path: {p}"

with open(DATA_ROOT / "class_mapping.json", "r", encoding="utf-8") as f:
    class_mapping = json.load(f)

print("class_mapping.json keys:", class_mapping.keys())

assert int(class_mapping["background_id"]) == 0, "background_id phải = 0"
assert (
    int(class_mapping["num_foreground_classes"]) == 76
), "num_foreground_classes phải = 76"
assert int(class_mapping["num_classes"]) == 77, "num_classes phải = 77"

CLASS_TO_ID = class_mapping["class_to_id"]  # source_class -> 1..76
TARGET_TITLE_TO_ID = class_mapping.get("target_title_to_id", {})
ID_TO_CLASS = {int(k): v for k, v in class_mapping["id_to_class"].items()}

BACKGROUND_ID = int(class_mapping["background_id"])
NUM_CLASSES = int(class_mapping["num_classes"])

print("BACKGROUND_ID:", BACKGROUND_ID)
print("NUM_CLASSES:", NUM_CLASSES)
print("Example classes:", [ID_TO_CLASS[i] for i in range(1, min(10, NUM_CLASSES))])
print("Dropped classes:", class_mapping.get("dropped_source_classes", []))

# %% [markdown]
# # Discover file stems + split train/val cố định

# %%
IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def find_image_path(img_dir: Path, stem: str):
    for ext in IMG_EXTS:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def list_image_stems(img_dir: Path):
    stems = []
    for p in img_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            stems.append(p.stem)
    return sorted(stems)


def discover_common_stems(split: str):
    img_dir = DATA_ROOT / split / "img"
    mask_dir = DATA_ROOT / split / "mask"
    img_stems = set(list_image_stems(img_dir))
    mask_stems = set([p.stem for p in mask_dir.glob("*.png")])
    return sorted(img_stems & mask_stems)


all_train_stems = discover_common_stems("train")
test_stems = discover_common_stems("test")

assert len(all_train_stems) > 0, "Không có train stems"
assert len(test_stems) > 0, "Không có test stems"

split_json_path = PROJECT_DIR / CONFIG["SPLIT_JSON_NAME"]

if split_json_path.exists():
    with open(split_json_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)
    train_stems = split_data["train"]
    val_stems = split_data["val"]
    print("Loaded fixed split from:", split_json_path)
else:
    rng = random.Random(CONFIG["SEED"])
    temp = all_train_stems.copy()
    rng.shuffle(temp)
    n_val = max(1, int(len(temp) * CONFIG["VAL_RATIO"]))
    val_stems = sorted(temp[:n_val])
    train_stems = sorted(temp[n_val:])

    split_data = {
        "seed": CONFIG["SEED"],
        "train": train_stems,
        "val": val_stems,
        "all_train_count": len(all_train_stems),
        "test_count": len(test_stems),
    }
    with open(split_json_path, "w", encoding="utf-8") as f:
        json.dump(split_data, f, ensure_ascii=False, indent=2)
    print("Saved fixed split to:", split_json_path)

print(f"train={len(train_stems)} | val={len(val_stems)} | test={len(test_stems)}")

# %% [markdown]
# # Tính class weights từ mask train


# %%
def compute_class_weights(
    mask_dir: Path, stems, num_classes: int, background_id: int = 0
):
    pixel_counter = np.zeros(num_classes, dtype=np.int64)

    for stem in tqdm(stems, desc="Compute class weights"):
        mask = np.array(Image.open(mask_dir / f"{stem}.png"), dtype=np.uint8)
        valid = mask[(mask >= 0) & (mask < num_classes)]
        counts = np.bincount(valid, minlength=num_classes)
        pixel_counter += counts

    freq = pixel_counter / (pixel_counter.sum() + 1e-12)
    weights = 1.0 / np.sqrt(freq + 1e-12)
    weights[pixel_counter == 0] = 0.0

    # giảm background weight
    weights[background_id] *= 0.5

    nonzero = weights[weights > 0]
    if len(nonzero) > 0:
        weights = weights / nonzero.mean()

    return torch.tensor(weights, dtype=torch.float32), pixel_counter


class_weights, pixel_counter = compute_class_weights(
    DATA_ROOT / "train" / "mask", train_stems, NUM_CLASSES, BACKGROUND_ID
)

print("background weight:", float(class_weights[BACKGROUND_ID]))
print("nonzero weights:", int((class_weights > 0).sum().item()))

top_ids = np.argsort(-pixel_counter)[:10]
print("\nTop pixel classes:")
for cid in top_ids:
    cname = "background" if cid == 0 else ID_TO_CLASS.get(int(cid), f"class_{cid}")
    print(f"id={cid:>2} | {cname:<20} | pixels={int(pixel_counter[cid])}")

# %% [markdown]
# # Augmentation + Dataset + DataLoader

# %%
SIZE = CONFIG["IMAGE_SIZE"]
BG = BACKGROUND_ID

train_tf = A.Compose(
    [
        A.LongestMaxSize(max_size=int(SIZE * 1.5), interpolation=cv2.INTER_LINEAR),
        A.PadIfNeeded(
            min_height=SIZE,
            min_width=SIZE,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=BG,
        ),
        A.RandomScale(
            scale_limit=(-0.5, 0.5),
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            p=0.8,
        ),
        A.PadIfNeeded(
            min_height=SIZE,
            min_width=SIZE,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=BG,
        ),
        A.RandomCrop(height=SIZE, width=SIZE),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.4),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(transpose_mask=False),
    ]
)

val_tf = A.Compose(
    [
        A.LongestMaxSize(max_size=SIZE, interpolation=cv2.INTER_LINEAR),
        A.PadIfNeeded(
            min_height=SIZE,
            min_width=SIZE,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=BG,
        ),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(transpose_mask=False),
    ]
)


class FoodSegRebalancedDataset(Dataset):
    def __init__(self, root, split, stems, transform=None):
        self.root = Path(root)
        self.split = split
        self.stems = list(stems)
        self.transform = transform
        self.img_dir = self.root / split / "img"
        self.mask_dir = self.root / split / "mask"

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]
        img_path = find_image_path(self.img_dir, stem)
        mask_path = self.mask_dir / f"{stem}.png"

        if img_path is None:
            raise FileNotFoundError(f"Không thấy ảnh cho stem={stem}")

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path), dtype=np.uint8)

        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask = aug["mask"]

        if not torch.is_tensor(mask):
            mask = torch.tensor(mask, dtype=torch.long)
        else:
            mask = mask.long()

        return {
            "image": image,
            "mask": mask,
            "stem": stem,
        }


train_ds = FoodSegRebalancedDataset(DATA_ROOT, "train", train_stems, transform=train_tf)
val_ds = FoodSegRebalancedDataset(DATA_ROOT, "train", val_stems, transform=val_tf)
test_ds = FoodSegRebalancedDataset(DATA_ROOT, "test", test_stems, transform=val_tf)

train_loader = DataLoader(
    train_ds,
    batch_size=CONFIG["BATCH_SIZE"],
    shuffle=True,
    num_workers=CONFIG["NUM_WORKERS"],
    pin_memory=CONFIG["PIN_MEMORY"],
    drop_last=True,
    persistent_workers=(CONFIG["NUM_WORKERS"] > 0),
)

val_loader = DataLoader(
    val_ds,
    batch_size=CONFIG["BATCH_SIZE"],
    shuffle=False,
    num_workers=CONFIG["NUM_WORKERS"],
    pin_memory=CONFIG["PIN_MEMORY"],
    drop_last=False,
    persistent_workers=(CONFIG["NUM_WORKERS"] > 0),
)

print("train_ds:", len(train_ds))
print("val_ds:", len(val_ds))
print("test_ds:", len(test_ds))

sample = train_ds[0]
print("image shape:", tuple(sample["image"].shape))
print("mask shape :", tuple(sample["mask"].shape))
print("mask min/max:", int(sample["mask"].min()), int(sample["mask"].max()))

# %% [markdown]
# # MODEL
# BiSeNet + ResNet18 pretrained

# %%
# =========================
# MODEL - CHANGED
# BiSeNetV1 + ResNet18 pretrained backbone
# =========================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SpatialPath(nn.Module):
    # output stride 8
    def __init__(self, out_ch=128):
        super().__init__()
        self.conv1 = ConvBNReLU(3, 64, k=7, s=2, p=3)  # /2
        self.conv2 = ConvBNReLU(64, 64, k=3, s=2, p=1)  # /4
        self.conv3 = ConvBNReLU(64, 64, k=3, s=2, p=1)  # /8
        self.conv_out = ConvBNReLU(64, out_ch, k=1, s=1, p=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv_out(x)
        return x


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBNReLU(in_ch, out_ch, k=3, s=1, p=1)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.Sigmoid(),
        )

    def forward(self, x):
        feat = self.conv(x)
        attn = self.attn(feat)
        return feat * attn


class FeatureFusionModule(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.convblk = ConvBNReLU(in_ch, out_ch, k=1, s=1, p=0)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, out_ch // 4, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch // 4, out_ch, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, fsp, fcp):
        feat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(feat)
        attn = self.attn(feat)
        feat_attn = feat * attn
        feat_out = feat + feat_attn
        return feat_out


class SegHead(nn.Module):
    def __init__(self, in_ch, mid_ch, n_classes):
        super().__init__()
        self.conv = ConvBNReLU(in_ch, mid_ch, k=3, s=1, p=1)
        self.dropout = nn.Dropout2d(0.1)
        self.cls = nn.Conv2d(mid_ch, n_classes, kernel_size=1)

    def forward(self, x, out_size):
        x = self.conv(x)
        x = self.dropout(x)
        x = self.cls(x)
        x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        return x


class ContextPathResNet18(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1  # /4
        self.layer2 = backbone.layer2  # /8
        self.layer3 = backbone.layer3  # /16
        self.layer4 = backbone.layer4  # /32

        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)

        self.global_context_conv = ConvBNReLU(512, 128, k=1, s=1, p=0)
        self.conv_head16 = ConvBNReLU(128, 128, k=3, s=1, p=1)
        self.conv_head32 = ConvBNReLU(128, 128, k=3, s=1, p=1)

    def forward(self, x):
        x = self.conv1(x)  # /2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # /4

        x = self.layer1(x)  # /4
        feat8 = self.layer2(x)  # /8
        feat16 = self.layer3(feat8)  # /16
        feat32 = self.layer4(feat16)  # /32

        gc = F.adaptive_avg_pool2d(feat32, 1)
        gc = self.global_context_conv(gc)
        gc = F.interpolate(
            gc, size=feat32.shape[-2:], mode="bilinear", align_corners=False
        )

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + gc
        feat32_up = F.interpolate(
            feat32_sum, size=feat16.shape[-2:], mode="bilinear", align_corners=False
        )
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(
            feat16_sum, size=feat8.shape[-2:], mode="bilinear", align_corners=False
        )
        feat16_up = self.conv_head16(feat16_up)

        # return context at /8 and aux source /16
        return feat8, feat16_up, feat32_up


class BiSeNetResNet18(nn.Module):
    def __init__(self, num_classes=77, pretrained_backbone=True, use_aux=True):
        super().__init__()
        self.use_aux = use_aux

        self.spatial_path = SpatialPath(out_ch=128)
        self.context_path = ContextPathResNet18(pretrained=pretrained_backbone)
        self.ffm = FeatureFusionModule(in_ch=128 + 128, out_ch=256)

        self.head = SegHead(256, 256, num_classes)

        if self.use_aux:
            self.aux_head16 = SegHead(128, 128, num_classes)
            self.aux_head32 = SegHead(128, 128, num_classes)

    def forward(self, x):
        out_size = x.shape[-2:]

        feat_sp = self.spatial_path(x)
        feat8, feat_cp8, feat_cp16 = self.context_path(x)

        feat_fuse = self.ffm(feat_sp, feat_cp8)
        logits = self.head(feat_fuse, out_size)

        if self.training and self.use_aux:
            aux16 = self.aux_head16(feat_cp8, out_size)
            aux32 = self.aux_head32(feat_cp16, out_size)
            return logits, aux16, aux32

        return logits


# %% [markdown]
# checkpoint helpers

# %%
# =========================
# CHECKPOINT HELPERS - CHANGED
# Giữ tối thiểu: latest + best
# Không lưu periodic để đỡ tốn disk/RAM serialize
# =========================
from pathlib import Path
import torch
import pandas as pd

PROJECT_DIR = Path(CONFIG["PROJECT_DIR"])
CKPT_DIR = PROJECT_DIR / CONFIG["CKPT_DIRNAME"]
PROJECT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)


def get_latest_checkpoint(ckpt_dir: Path):
    latest = ckpt_dir / "latest.pth"
    return latest if latest.exists() else None


def save_checkpoint(state: dict, ckpt_dir: Path, is_best: bool = False):
    torch.save(state, ckpt_dir / "latest.pth")
    if is_best and CONFIG["SAVE_BEST"]:
        torch.save(state, ckpt_dir / "best.pth")


def append_log_row(log_path: Path, row: dict):
    df = pd.DataFrame([row])
    if log_path.exists():
        df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        df.to_csv(log_path, index=False)


# %% [markdown]
# # train/val functions

# %%
# =========================
# TRAIN / VAL FUNCTIONS - CHANGED
# phù hợp output của BiSeNet ResNet18 pretrained
# =========================
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F


@torch.no_grad()
def update_hist(hist, pred, target, num_classes):
    mask = (target >= 0) & (target < num_classes)
    x = num_classes * target[mask] + pred[mask]
    binc = torch.bincount(x.reshape(-1), minlength=num_classes**2)
    hist += binc.reshape(num_classes, num_classes)
    return hist


@torch.no_grad()
def compute_metrics_from_hist(hist):
    hist = hist.float()
    eps = 1e-6
    tp = torch.diag(hist)
    gt = hist.sum(dim=1)
    pred = hist.sum(dim=0)

    iou = tp / (gt + pred - tp + eps)
    acc_cls = tp / (gt + eps)
    aacc = tp.sum() / (hist.sum() + eps)

    return {
        "mIoU": torch.nanmean(iou).item(),
        "mAcc": torch.nanmean(acc_cls).item(),
        "aAcc": aacc.item(),
    }


def train_one_epoch(
    model, loader, criterion, optimizer, scheduler, scaler, device, epoch
):
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [Train]", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            "cuda", enabled=(device.type == "cuda" and CONFIG["USE_AMP"])
        ):
            outputs = model(images)

            if CONFIG["USE_AUX_HEAD"]:
                main_logits, aux16, aux32 = outputs
                loss_main = criterion(main_logits, masks)
                loss_aux16 = criterion(aux16, masks)
                loss_aux32 = criterion(aux32, masks)
                loss = (
                    loss_main + CONFIG["AUX_WEIGHT"] * (loss_aux16 + loss_aux32) * 0.5
                )
            else:
                main_logits = outputs
                loss = criterion(main_logits, masks)

        scaler.scale(loss).backward()

        if CONFIG["GRAD_CLIP_NORM"] is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["GRAD_CLIP_NORM"])

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += float(loss.item())
        pbar.set_postfix(
            loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}"
        )

        del images, masks, outputs, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return total_loss / max(1, len(loader))


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device, num_classes, epoch):
    model.eval()
    total_loss = 0.0
    hist = torch.zeros((num_classes, num_classes), device=device, dtype=torch.int64)

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [Val]", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True).long()

        with torch.amp.autocast(
            "cuda", enabled=(device.type == "cuda" and CONFIG["USE_AMP"])
        ):
            logits = model(images)
            loss = criterion(logits, masks)

        total_loss += float(loss.item())
        preds = torch.argmax(logits, dim=1)
        hist = update_hist(hist, preds, masks, num_classes)

        del images, masks, logits, preds
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metrics = compute_metrics_from_hist(hist)
    return total_loss / max(1, len(loader)), metrics


# %% [markdown]
# init model / optimizer / resume

# %%
# =========================
# INIT MODEL / OPT / RESUME - CHANGED
# =========================
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

model = BiSeNetResNet18(
    num_classes=NUM_CLASSES,
    pretrained_backbone=CONFIG["PRETRAINED_BACKBONE"],
    use_aux=CONFIG["USE_AUX_HEAD"],
).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=CONFIG["LR"],
    momentum=CONFIG["MOMENTUM"],
    weight_decay=CONFIG["WEIGHT_DECAY"],
)

total_iters = CONFIG["EPOCHS"] * max(1, len(train_loader))
poly_lambda = (
    lambda it: (1 - min(it, total_iters) / max(1, total_iters)) ** CONFIG["POLY_POWER"]
)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_lambda)

scaler = torch.amp.GradScaler(
    "cuda", enabled=(device.type == "cuda" and CONFIG["USE_AMP"])
)

start_epoch = 1
best_miou = -1.0
global_iter = 0

latest_ckpt = get_latest_checkpoint(CKPT_DIR)

if CONFIG["RESUME"] and latest_ckpt is not None:
    print("Resuming from:", latest_ckpt)
    ckpt = torch.load(latest_ckpt, map_location=device)

    model.load_state_dict(ckpt["model_state"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer_state"])

    if ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    if ckpt.get("scaler_state") is not None and device.type == "cuda":
        scaler.load_state_dict(ckpt["scaler_state"])

    start_epoch = int(ckpt["epoch"]) + 1
    best_miou = float(ckpt.get("best_miou", -1.0))
    global_iter = int(ckpt.get("global_iter", 0))

    print(f"Resume OK -> start_epoch={start_epoch}, best_miou={best_miou:.4f}")
else:
    print("No checkpoint found. Train from scratch.")

batch = next(iter(train_loader))
print("sanity image:", tuple(batch["image"].shape))
print("sanity mask :", tuple(batch["mask"].shape))
print("mask min/max:", int(batch["mask"].min()), int(batch["mask"].max()))

# %%
# =========================
# ALL-IN-ONE PRACTICAL PATCH
# Base on your old notebook, but safer for current dataset behavior
# Run this cell AFTER:
#   - DATA_ROOT
#   - class_mapping loaded
#   - train_stems / val_stems ready
#   - find_image_path(...) already available (or this cell will redefine it)
# =========================

import os
import gc
import json
import math
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm.auto import tqdm

# -------------------------
# 1) PRACTICAL CONFIG RESET
# -------------------------
CONFIG.update(
    {
        # dataset
        "NUM_CLASSES": 77,
        "BACKGROUND_ID": 0,
        # training target
        "EPOCHS": 40,  # increase total target epochs
        "IMAGE_SIZE": 384,  # start safe; after stable -> try 448
        "BATCH_SIZE": 4,  # if OOM -> 2
        "NUM_WORKERS": 2,  # if RAM issue -> 0
        "PIN_MEMORY": True,
        # optimizer / schedule
        "LR": 0.005,  # gentler than 0.01 for pretrained backbone
        "MOMENTUM": 0.9,
        "WEIGHT_DECAY": 5e-4,
        "POLY_POWER": 0.9,
        # model
        "PRETRAINED_BACKBONE": True,
        "USE_AUX_HEAD": True,
        "AUX_WEIGHT": 0.3,
        # mixed precision
        "USE_AMP": True,
        # resume
        "RESUME": True,
        "SAVE_BEST": True,
        "SAVE_LATEST_EVERY_EPOCH": True,
        # stability
        "GRAD_CLIP_NORM": 1.0,
        # new practical knobs
        "USE_DICE_LOSS": True,
        "DICE_WEIGHT": 0.5,
        "USE_WEIGHTED_CE": True,
        "CLASS_WEIGHT_CLIP_MIN": 0.5,
        "CLASS_WEIGHT_CLIP_MAX": 3.0,
        # data improvement
        "USE_FOREGROUND_AWARE_CROP": True,
        "FG_CROP_MIN_RATIO": 0.03,
        "USE_WEIGHTED_SAMPLER": False,  # set True only if rare-class presence is terrible
    }
)

# -------------------------
# 2) PATHS
# -------------------------
PROJECT_DIR = Path(CONFIG["DRIVE_PROJECT_DIR"])
CKPT_DIR = PROJECT_DIR / CONFIG["CKPT_DIRNAME"]
PROJECT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_WEIGHTS_PATH = PROJECT_DIR / "class_weights.pt"
LOG_CSV_PATH = PROJECT_DIR / CONFIG["LOG_CSV_NAME"]

print("PROJECT_DIR:", PROJECT_DIR)
print("CKPT_DIR   :", CKPT_DIR)

# -------------------------
# 3) FALLBACK HELPERS
# -------------------------
IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def find_image_path_local(img_dir: Path, stem: str):
    for ext in IMG_EXTS:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


try:
    find_image_path
except NameError:
    find_image_path = find_image_path_local


# -------------------------
# 4) CLASS WEIGHTS
#    - load from Drive if exists
#    - else compute once, save to Drive
#    - then CLIP to avoid extreme imbalance hurting optimization
# -------------------------
def compute_class_weights(
    mask_dir: Path, stems, num_classes: int, background_id: int = 0
):
    pixel_counter = np.zeros(num_classes, dtype=np.int64)

    for stem in tqdm(stems, desc="Compute class weights"):
        mask = np.array(Image.open(mask_dir / f"{stem}.png"), dtype=np.uint8)
        valid = mask[(mask >= 0) & (mask < num_classes)]
        counts = np.bincount(valid, minlength=num_classes)
        pixel_counter += counts

    freq = pixel_counter / (pixel_counter.sum() + 1e-12)
    weights = 1.0 / np.sqrt(freq + 1e-12)
    weights[pixel_counter == 0] = 0.0

    # background down-weight
    weights[background_id] *= 0.5

    nonzero = weights[weights > 0]
    if len(nonzero) > 0:
        weights = weights / nonzero.mean()

    return torch.tensor(weights, dtype=torch.float32), pixel_counter


if CLASS_WEIGHTS_PATH.exists():
    class_weights = torch.load(CLASS_WEIGHTS_PATH, map_location="cpu")
    print("Loaded class_weights from:", CLASS_WEIGHTS_PATH)
    pixel_counter = None
else:
    class_weights, pixel_counter = compute_class_weights(
        DATA_ROOT / "train" / "mask",
        train_stems,
        CONFIG["NUM_CLASSES"],
        CONFIG["BACKGROUND_ID"],
    )
    torch.save(class_weights.cpu(), CLASS_WEIGHTS_PATH)
    print("Saved class_weights to:", CLASS_WEIGHTS_PATH)

if CONFIG["USE_WEIGHTED_CE"]:
    class_weights = torch.clamp(
        class_weights,
        min=CONFIG["CLASS_WEIGHT_CLIP_MIN"],
        max=CONFIG["CLASS_WEIGHT_CLIP_MAX"],
    )

# keep background modest, not too tiny
class_weights[CONFIG["BACKGROUND_ID"]] = max(
    float(class_weights[CONFIG["BACKGROUND_ID"]]), 0.5
)

print("class_weights min/max:", float(class_weights.min()), float(class_weights.max()))
print("background weight after clip:", float(class_weights[CONFIG["BACKGROUND_ID"]]))


# -------------------------
# 5) FOREGROUND-AWARE CROP
# -------------------------
def foreground_aware_crop(
    image, mask, crop_size=384, background_id=0, min_fg_ratio=0.03, max_tries=10
):
    h, w = mask.shape
    ch, cw = crop_size, crop_size

    if h < ch or w < cw:
        pad_h = max(0, ch - h)
        pad_w = max(0, cw - w)
        image = cv2.copyMakeBorder(
            image, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        mask = cv2.copyMakeBorder(
            mask,
            0,
            pad_h,
            0,
            pad_w,
            borderType=cv2.BORDER_CONSTANT,
            value=background_id,
        )
        h, w = mask.shape

    ys, xs = np.where(mask != background_id)

    if len(xs) > 0:
        for _ in range(max_tries):
            idx = np.random.randint(0, len(xs))
            cx, cy = xs[idx], ys[idx]

            x1 = int(np.clip(cx - cw // 2, 0, w - cw))
            y1 = int(np.clip(cy - ch // 2, 0, h - ch))

            crop_img = image[y1 : y1 + ch, x1 : x1 + cw]
            crop_mask = mask[y1 : y1 + ch, x1 : x1 + cw]

            if (crop_mask != background_id).mean() >= min_fg_ratio:
                return crop_img, crop_mask

    x1 = np.random.randint(0, max(1, w - cw + 1))
    y1 = np.random.randint(0, max(1, h - ch + 1))
    return image[y1 : y1 + ch, x1 : x1 + cw], mask[y1 : y1 + ch, x1 : x1 + cw]


# -------------------------
# 6) AUGMENTATION
#    lighter than old version
# -------------------------
SIZE = CONFIG["IMAGE_SIZE"]
BG = CONFIG["BACKGROUND_ID"]

train_tf = A.Compose(
    [
        A.LongestMaxSize(max_size=int(SIZE * 1.25), interpolation=cv2.INTER_LINEAR),
        A.PadIfNeeded(
            min_height=SIZE,
            min_width=SIZE,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=BG,
        ),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(
            brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03, p=0.35
        ),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(transpose_mask=False),
    ]
)

val_tf = A.Compose(
    [
        A.LongestMaxSize(max_size=SIZE, interpolation=cv2.INTER_LINEAR),
        A.PadIfNeeded(
            min_height=SIZE,
            min_width=SIZE,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=BG,
        ),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(transpose_mask=False),
    ]
)


# -------------------------
# 7) DATASET
# -------------------------
class FoodSegRebalancedDatasetPractical(Dataset):
    def __init__(self, root, split, stems, transform=None, train_mode=False):
        self.root = Path(root)
        self.split = split
        self.stems = list(stems)
        self.transform = transform
        self.train_mode = train_mode
        self.img_dir = self.root / split / "img"
        self.mask_dir = self.root / split / "mask"

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]
        img_path = find_image_path(self.img_dir, stem)
        mask_path = self.mask_dir / f"{stem}.png"

        if img_path is None:
            raise FileNotFoundError(f"Cannot find image for stem={stem}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Cannot find mask for stem={stem}")

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path), dtype=np.uint8)

        # foreground-aware crop BEFORE albumentations normalize/tensor
        if self.train_mode and CONFIG["USE_FOREGROUND_AWARE_CROP"]:
            image, mask = foreground_aware_crop(
                image=image,
                mask=mask,
                crop_size=CONFIG["IMAGE_SIZE"],
                background_id=CONFIG["BACKGROUND_ID"],
                min_fg_ratio=CONFIG["FG_CROP_MIN_RATIO"],
                max_tries=10,
            )

        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask = aug["mask"]

        if not torch.is_tensor(mask):
            mask = torch.tensor(mask, dtype=torch.long)
        else:
            mask = mask.long()

        return {
            "image": image,
            "mask": mask,
            "stem": stem,
        }


train_ds = FoodSegRebalancedDatasetPractical(
    DATA_ROOT, "train", train_stems, transform=train_tf, train_mode=True
)
val_ds = FoodSegRebalancedDatasetPractical(
    DATA_ROOT, "train", val_stems, transform=val_tf, train_mode=False
)

# optional sampler: by foreground ratio proxy from masks
sampler = None
shuffle = True

if CONFIG["USE_WEIGHTED_SAMPLER"]:
    print("Building weighted sampler...")
    sample_weights = []
    for stem in tqdm(train_stems, desc="Sampler weights"):
        mask = np.array(
            Image.open(DATA_ROOT / "train" / "mask" / f"{stem}.png"), dtype=np.uint8
        )
        fg_ratio = float((mask != CONFIG["BACKGROUND_ID"]).mean())
        # more weight for images with meaningful foreground
        w = 1.0 + min(4.0, fg_ratio * 10.0)
        sample_weights.append(w)
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )
    shuffle = False

train_loader = DataLoader(
    train_ds,
    batch_size=CONFIG["BATCH_SIZE"],
    shuffle=shuffle if sampler is None else False,
    sampler=sampler,
    num_workers=CONFIG["NUM_WORKERS"],
    pin_memory=CONFIG["PIN_MEMORY"],
    drop_last=True,
    persistent_workers=(CONFIG["NUM_WORKERS"] > 0),
)

val_loader = DataLoader(
    val_ds,
    batch_size=CONFIG["BATCH_SIZE"],
    shuffle=False,
    num_workers=CONFIG["NUM_WORKERS"],
    pin_memory=CONFIG["PIN_MEMORY"],
    drop_last=False,
    persistent_workers=(CONFIG["NUM_WORKERS"] > 0),
)

print("train_ds:", len(train_ds))
print("val_ds  :", len(val_ds))

sample = train_ds[0]
print("sample image:", tuple(sample["image"].shape))
print("sample mask :", tuple(sample["mask"].shape))
print("mask min/max:", int(sample["mask"].min()), int(sample["mask"].max()))


# -------------------------
# 8) LOSS
#    CrossEntropy + Dice
# -------------------------
class SoftDiceLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=None, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits, target):
        # logits: [B, C, H, W]
        # target: [B, H, W]
        probs = torch.softmax(logits, dim=1)

        target_clamped = target.clone()
        if self.ignore_index is not None:
            target_clamped[target_clamped == self.ignore_index] = 0

        one_hot = (
            F.one_hot(target_clamped, num_classes=self.num_classes)
            .permute(0, 3, 1, 2)
            .float()
        )

        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index).unsqueeze(1)
            probs = probs * valid_mask
            one_hot = one_hot * valid_mask

        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dims)
        cardinality = probs.sum(dims) + one_hot.sum(dims)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class SegLoss(nn.Module):
    def __init__(self, ce_weight=None, num_classes=77, dice_weight=0.5, use_dice=True):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=ce_weight)
        self.dice = SoftDiceLoss(num_classes=num_classes)
        self.dice_weight = dice_weight
        self.use_dice = use_dice

    def forward(self, logits, target):
        ce_loss = self.ce(logits, target)
        if not self.use_dice:
            return ce_loss
        dice_loss = self.dice(logits, target)
        return ce_loss + self.dice_weight * dice_loss


# -------------------------
# 9) CHECKPOINT HELPERS
# -------------------------
def get_latest_checkpoint(ckpt_dir: Path):
    latest = ckpt_dir / "latest.pth"
    return latest if latest.exists() else None


def save_checkpoint(state: dict, ckpt_dir: Path, is_best: bool = False):
    torch.save(state, ckpt_dir / "latest.pth")
    if is_best and CONFIG["SAVE_BEST"]:
        torch.save(state, ckpt_dir / "best.pth")


def append_log_row(log_path: Path, row: dict):
    df = pd.DataFrame([row])
    if log_path.exists():
        df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        df.to_csv(log_path, index=False)


# -------------------------
# 10) METRICS
# -------------------------
@torch.no_grad()
def update_hist(hist, pred, target, num_classes):
    mask = (target >= 0) & (target < num_classes)
    x = num_classes * target[mask] + pred[mask]
    binc = torch.bincount(x.reshape(-1), minlength=num_classes**2)
    hist += binc.reshape(num_classes, num_classes)
    return hist


@torch.no_grad()
def compute_metrics_from_hist(hist):
    hist = hist.float()
    eps = 1e-6
    tp = torch.diag(hist)
    gt = hist.sum(dim=1)
    pred = hist.sum(dim=0)

    iou = tp / (gt + pred - tp + eps)
    acc_cls = tp / (gt + eps)
    aacc = tp.sum() / (hist.sum() + eps)

    return {
        "mIoU": torch.nanmean(iou).item(),
        "mAcc": torch.nanmean(acc_cls).item(),
        "aAcc": aacc.item(),
    }


# -------------------------
# 11) TRAIN / VAL
# -------------------------
def train_one_epoch(
    model, loader, criterion, optimizer, scheduler, scaler, device, epoch
):
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [Train]", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            "cuda", enabled=(device.type == "cuda" and CONFIG["USE_AMP"])
        ):
            outputs = model(images)

            if CONFIG["USE_AUX_HEAD"]:
                main_logits, aux16, aux32 = outputs
                loss_main = criterion(main_logits, masks)
                loss_aux16 = criterion(aux16, masks)
                loss_aux32 = criterion(aux32, masks)
                loss = (
                    loss_main + CONFIG["AUX_WEIGHT"] * (loss_aux16 + loss_aux32) * 0.5
                )
            else:
                main_logits = outputs
                loss = criterion(main_logits, masks)

        scaler.scale(loss).backward()

        if CONFIG["GRAD_CLIP_NORM"] is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["GRAD_CLIP_NORM"])

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += float(loss.item())
        pbar.set_postfix(
            loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}"
        )

        del images, masks, outputs, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return total_loss / max(1, len(loader))


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device, num_classes, epoch):
    model.eval()
    total_loss = 0.0
    hist = torch.zeros((num_classes, num_classes), device=device, dtype=torch.int64)

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [Val]", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True).long()

        with torch.amp.autocast(
            "cuda", enabled=(device.type == "cuda" and CONFIG["USE_AMP"])
        ):
            logits = model(images)
            loss = criterion(logits, masks)

        total_loss += float(loss.item())
        preds = torch.argmax(logits, dim=1)
        hist = update_hist(hist, preds, masks, num_classes)

        del images, masks, logits, preds
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metrics = compute_metrics_from_hist(hist)
    return total_loss / max(1, len(loader)), metrics


# -------------------------
# 12) INIT MODEL / OPT / RESUME
#    assumes BiSeNetResNet18 class is already defined from your old code
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

model = BiSeNetResNet18(
    num_classes=CONFIG["NUM_CLASSES"],
    pretrained_backbone=CONFIG["PRETRAINED_BACKBONE"],
    use_aux=CONFIG["USE_AUX_HEAD"],
).to(device)

criterion = SegLoss(
    ce_weight=class_weights.to(device) if CONFIG["USE_WEIGHTED_CE"] else None,
    num_classes=CONFIG["NUM_CLASSES"],
    dice_weight=CONFIG["DICE_WEIGHT"],
    use_dice=CONFIG["USE_DICE_LOSS"],
)

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=CONFIG["LR"],
    momentum=CONFIG["MOMENTUM"],
    weight_decay=CONFIG["WEIGHT_DECAY"],
)

total_iters = CONFIG["EPOCHS"] * max(1, len(train_loader))
poly_lambda = (
    lambda it: (1 - min(it, total_iters) / max(1, total_iters)) ** CONFIG["POLY_POWER"]
)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_lambda)

scaler = torch.amp.GradScaler(
    "cuda", enabled=(device.type == "cuda" and CONFIG["USE_AMP"])
)

start_epoch = 1
best_miou = -1.0
global_iter = 0

latest_ckpt = get_latest_checkpoint(CKPT_DIR)

if CONFIG["RESUME"] and latest_ckpt is not None:
    print("Resuming from:", latest_ckpt)
    ckpt = torch.load(latest_ckpt, map_location=device)

    model.load_state_dict(ckpt["model_state"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer_state"])

    if ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    if ckpt.get("scaler_state") is not None and device.type == "cuda":
        scaler.load_state_dict(ckpt["scaler_state"])

    start_epoch = int(ckpt["epoch"]) + 1
    best_miou = float(ckpt.get("best_miou", -1.0))
    global_iter = int(ckpt.get("global_iter", 0))

    print(f"Resume OK -> start_epoch={start_epoch}, best_miou={best_miou:.4f}")
else:
    print("No checkpoint found. Train from scratch.")

if start_epoch > CONFIG["EPOCHS"]:
    print(f"WARNING: start_epoch={start_epoch} > EPOCHS={CONFIG['EPOCHS']}")
    print("=> increase CONFIG['EPOCHS'] if you want to continue training.")

# -------------------------
# 13) TRAIN LOOP
# -------------------------
for epoch in range(start_epoch, CONFIG["EPOCHS"] + 1):
    train_loss = train_one_epoch(
        model, train_loader, criterion, optimizer, scheduler, scaler, device, epoch
    )
    val_loss, metrics = validate_one_epoch(
        model, val_loader, criterion, device, CONFIG["NUM_CLASSES"], epoch
    )
    global_iter += len(train_loader)

    miou = metrics["mIoU"]
    is_best = miou > best_miou
    if is_best:
        best_miou = miou

    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict() if device.type == "cuda" else None,
        "best_miou": best_miou,
        "global_iter": global_iter,
        "config_snapshot": dict(CONFIG),
    }

    if CONFIG["SAVE_LATEST_EVERY_EPOCH"]:
        save_checkpoint(state, CKPT_DIR, is_best=is_best)

    row = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "mIoU": metrics["mIoU"],
        "mAcc": metrics["mAcc"],
        "aAcc": metrics["aAcc"],
        "best_mIoU": best_miou,
        "lr": optimizer.param_groups[0]["lr"],
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "image_size": CONFIG["IMAGE_SIZE"],
        "batch_size": CONFIG["BATCH_SIZE"],
        "dice_weight": CONFIG["DICE_WEIGHT"],
    }
    append_log_row(LOG_CSV_PATH, row)

    print(
        f"Epoch {epoch:03d} | "
        f"train_loss={train_loss:.4f} | "
        f"val_loss={val_loss:.4f} | "
        f"mIoU={metrics['mIoU']:.4f} | "
        f"mAcc={metrics['mAcc']:.4f} | "
        f"aAcc={metrics['aAcc']:.4f} | "
        f"best={best_miou:.4f}"
    )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\nDone.")
print("Latest:", CKPT_DIR / "latest.pth")
print("Best  :", CKPT_DIR / "best.pth")
print("Log   :", LOG_CSV_PATH)
