# Cấu hình dataset **FoodSeg103 rebalanced** (train)

Thư mục tài liệu này mô tả **taxonomy và metadata** của bản dữ liệu tại `source/dataset/foodseg103_rebalanced/` (hoặc `foodseg103_rebalanced.zip`). Dùng khi huấn luyện **không** còn 103 lớp gốc mà đã **gộp lớp hiếm**, **bỏ một số lớp dessert / kelp** để giảm mất cân bằng cực đoan.

---

## Cấu trúc thư mục dữ liệu

```
foodseg103_rebalanced/
├── class_mapping.csv           # nguồn sự thật remap (103 dòng nguồn)
├── class_rebalance_plan.csv    # thống kê merge/drop (benchmark)
├── class_mapping.json          # sinh bởi convertMask --rebalanced (dùng cho train code)
├── meta.json                   # ontology 76 lớp sau rebalance (cập nhật bởi script)
├── train/
│   ├── img/
│   ├── ann/                    # *.jpg.json
│   └── mask/                   # *.png — grayscale, pixel = 0 (bg) hoặc 1..76
└── test/
    ├── img/
    ├── ann/
    └── mask/
```

---

## Sinh mask + dọn dữ liệu hỏng

Dùng CLI trong `source/preprocessing/convertMask.py`:

```bash
python source/preprocessing/convertMask.py --rebalanced --clean --split both --overwrite \
  --data-root source/dataset/foodseg103_rebalanced
```

| Flag | Ý nghĩa |
|------|---------|
| `--rebalanced` | Đọc `class_mapping.csv`, id nền **0**, foreground **1..76**, ghi `class_mapping.json` + cập nhật `meta.json` (`classes` = 76 lớp). |
| `--clean` | Xóa cặp `img`/`ann` không khớp; JPEG không đọc được bằng Pillow; JSON hỏng; `mask` mồ côi. |
| `--overwrite` | Ghi đè toàn bộ PNG trong `mask/`. |
| `--delete-on-mask-error` | (Tuỳ chọn) Nếu raster một ảnh lỗi (bitmap decode, …) thì xóa luôn bộ ba img/ann/mask của stem đó. Mặc định **không** bật. |
| `--no-update-meta` | Giữ nguyên `meta.json` (chỉ sinh mask + JSON mapping). |

**Giá trị pixel mask:** `0` = background (không nhãn + lớp đã **drop** + pixel nền), `1..76` = lớp sau gộp (theo `new_class_id` trong CSV).

**`classTitle` trong JSON:** có thể là tên nguyên liệu gốc (`cilantro mint`) **hoặc** tên nhóm đã gộp (`vegetable`, `mushroom`, `seafood`, `nut`). Script map cả hai nhờ `class_to_id` + `target_title_to_id` trong `class_mapping.json`.

---

## Số lớp và logits mô hình

| Khái niệm | Giá trị |
|-----------|---------|
| Lớp nguồn (gốc FoodSeg103) | **103** `classTitle` |
| Lớp giữ / remap được (`kept=True`) | **94** dòng trong `class_mapping.csv` (gộp về **76** id) |
| **Lớp semantic sau rebalance** | **76** id **`1 … 76`** trên mask |
| **Background** | **`0`** trên mask → **`num_classes = 77`** (0..76) nếu bạn remap 0..76 liên tục cho CE; hoặc giữ pixel 1..76 + 0 và logits 77 kênh. |

Resize mask: **nearest neighbor**; chồng object: thứ tự trong `objects[]`, object sau đè object trước (giống `dataset.md`).

---

## File `class_mapping.json` (sau `--rebalanced`)

| Khóa | Ý nghĩa |
|------|---------|
| `schema` | `"foodseg103_rebalanced"` |
| `class_to_id` | Mỗi **source_class** (tên fine-grained) → id **1..76** |
| `target_title_to_id` | Tên nhóm (`vegetable`, …) → cùng id (để đọc JSON đã relabel nhóm) |
| `id_to_class` | id → `target_class` hiển thị |
| `background_id` | `0` |
| `num_foreground_classes` | `76` |
| `num_classes` | `77` (gồm background) |
| `dropped_source_classes` | 9 lớp nguồn bị drop |

---

## File `class_mapping.csv`

Cột: `source_class`, `target_class`, `kept`, `new_class_id`.

**Gộp nhóm chính:**

| Nhóm đích | `new_class_id` | Ví dụ nguồn |
|-----------|----------------|-------------|
| `vegetable` | 73 | cilantro mint, spring onion, garlic, … |
| `mushroom` | 40 | shiitake, enoki, … |
| `seafood` | 64 | shrimp, crab, … |
| `nut` | 42 | almond, peanut, … |

---

## File `class_rebalance_plan.csv`

Cột `action`: `keep` | `merge_keep` | `drop`. Các lớp **drop**: dessert/snack (`ice cream`, `cake`, …) và `kelp`.

---

## Quy mô split (sau clean trên bản repo)

| Split | `img` / `ann` / `mask` (cùng stem) |
|-------|-------------------------------------|
| `train` | **3981** (đã loại JPEG/JSON hỏng, ví dụ file rỗng) |
| `test` | **2135** |

Đếm lại: `(Get-ChildItem ...\train\img).Count` v.v.

---

## Liên kết

- Định dạng JSON / bitmap: `source/dataset/docs/dataset.md`
- Cùng file script: chế độ **không** `--rebalanced` = FoodSeg103 103 lớp + background 103.

---

## Checklist trước khi train

1. `DATA_ROOT` → `foodseg103_rebalanced`, có đủ `train/{img,ann,mask}` (và `test/...` nếu dùng).
2. Đọc nhãn từ `class_mapping.json` bản rebalanced, **không** lẫn `foodseg103/class_mapping.json` 103 lớp.
3. Paper: ghi **76 foreground + bg 0**, merge/drop theo CSV.
