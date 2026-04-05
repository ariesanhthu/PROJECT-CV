  # Dataset trong repo (`source/dataset/`)

  ## Cấu trúc thư mục

  ```
  source/dataset/
  ├── docs/
  │   ├── dataset.md          # file này
  │   └── train.md
  └── foodseg103/
      ├── README.md
      ├── LICENSE.md
      ├── meta.json           # metadata dự án Supervisely (lớp, tag, cấu hình)
      ├── class_mapping.json  # ánh xạ tên lớp training ↔ id (sinh bởi convertMask)
      ├── train/
      │   ├── img/            # ảnh RGB (.jpg)
      │   ├── ann/            # nhãn Supervisely JSON (mỗi ảnh một file)
      │   └── mask/           # mask PNG grayscale (sinh bởi preprocessing; 1–1 với img)
      └── test/
          ├── img/
          ├── ann/
          └── mask/            # (nếu đã chạy convertMask --split both)
  ```

  ### Thống kê bản copy hiện có (trên đĩa)

  | Split | `img/` | `ann/` | `mask/` | Ghi chú |
  |-------|--------|--------|---------|---------|
  | `train` | 922 | 922 | 922 (sau `convertMask`) | Cùng basename: `00000001.jpg` ↔ `00000001.png` |
  | `test` | 399 | 399 | tuỳ đã convert hay chưa | |

  **Lưu ý:** benchmark **FoodSeg103** trong paper thường báo **4.983 train / 2.135 test** và ~7.118 ảnh. Copy trong repo này là **subset nhỏ hơn** — khi so sánh số liệu với paper phải ghi rõ dùng full hay subset.

  Không có thư mục `val/` riêng: muốn validation thì **tách một phần từ `train`** (ví dụ 5–10%, seed cố định) hoặc dùng `test` chỉ cho báo cáo cuối (tránh tune trên test).

  ---

  ## Định dạng nhãn (JSON Supervisely)

  Mỗi ảnh `train/img/XXXXXXXX.jpg` có file tương ứng `train/ann/XXXXXXXX.jpg.json`.

  Cấu trúc chính:

  - `size.height`, `size.width`: kích thước **theo annotation** (trong mẫu thường 256×256; ảnh gốc có thể khác — **nên đọc kích thước thực từ ảnh** khi raster mask).
  - `objects[]`: danh sách instance, mỗi object:
    - `classTitle`: tên lớp nguyên liệu (string, ví dụ `strawberry`, `chicken duck`).
    - `geometryType`: `"bitmap"`.
    - `bitmap.data`: chuỗi base64 của **PNG nén** (zlib) — decode ra mask nhị phân cho instance đó.
    - `bitmap.origin`: `[x, y]` — góc trên-trái của bitmap trong **hệ tọa độ ảnh gốc** (hoặc theo `size` trong JSON, tùy pipeline; cần align với ảnh sau resize).

  **Số lớp:** trên toàn bộ `ann/` hiện có **103** giá trị `classTitle` khác nhau. Cho **semantic segmentation** thực tế thường dùng **104 channel** = **103 nguyên liệu + 1 background** (pixel không thuộc mask nào → background).

  **Chồng lấp:** nhiều object có thể overlap. Cách raster phổ biến: vẽ theo thứ tự (object sau **đè** object trước) hoặc quy ước priority theo lớp — **phải cố định và ghi trong báo cáo** để reproducible.

  **Decode `bitmap.data`:** trong export này thường là `base64(zlib(PNG))` — giải base64 rồi **zlib decompress** trước khi đọc PNG (xem `source/preprocessing/convertMask.py`).

  ---

  ## Thư mục `mask/` (nhãn đã raster)

  Sinh tự động bằng:

  `python source/preprocessing/convertMask.py` (mặc định split `train`; thêm `--split both --overwrite` cho cả test và ghi đè).

  - **Định dạng:** PNG một kênh (`PIL` mode `L`), `uint8`, cùng kích thước ảnh tương ứng trong `img/`.
  - **Giá trị pixel:** `0..102` = id nguyên liệu (theo `class_mapping.json`), **`103` = background** (pixel không thuộc object nào).
  - **Tên file:** `train/mask/00000001.png` tương ứng `train/img/00000001.jpg` (bỏ `.jpg` trong tên mask).

  Dùng trực tiếp cho DataLoader / MMSeg / Albumentations: load ảnh + mask, resize mask **nearest neighbor**.

  ---

  ## `foodseg103/class_mapping.json`

  File JSON do `convertMask.py` ghi (cập nhật mỗi lần chạy script với split tương ứng).

  | Khóa | Kiểu | Ý nghĩa |
  |------|------|---------|
  | `class_to_id` | `object` | `classTitle` (string) → `int` **0..102**, thứ tự **alphabetical** trên toàn bộ `classTitle` xuất hiện trong các split được quét (train hoặc train+test). |
  | `id_to_class` | `object` | Nghịch đảo: key là **chuỗi** `"0"`..`"102"` (JSON không cho key số thuần), value là tên lớp. |
  | `background_id` | `int` | Luôn **`103`** — pixel nền trên mask PNG. |
  | `num_ingredient_classes` | `int` | **`103`** — số lớp nguyên liệu (không tính background). |

  **Huấn luyện:** `num_classes = 104` (logits 0..103), loss thường dùng toàn bộ 104 hoặc bỏ qua `ignore_index` nếu có vùng ignore riêng.

  **Lưu ý:** Thứ tự id **không** trùng field `id` trong `meta.json` (id Supervisely). Luôn dùng `class_mapping.json` làm nguồn đúng cho training.

  ---

  ## `foodseg103/meta.json`

  Metadata kiểu **Supervisely / DatasetNinja**: định nghĩa **lớp ontology** và **tag** của project, không phải mask từng ảnh.

  Cấu trúc tổng quan (root object):

  | Khóa | Mô tả |
  |------|--------|
  | `classes` | Mảng các lớp có thể gán cho object. Mỗi phần tử gồm: `title` (tên lớp, khớp `classTitle` trong `ann/*.json`), `shape` (thường `"bitmap"`), `color` (hex UI), `geometry_config`, `id` (**ID nội bộ Supervisely**, số lớn — **không dùng làm label training**), `hotkey`. |
  | `tags` | Mảng tag phụ (nhóm món: `vegetable`, `meat`, …): `name`, `id`, `value_type`, `color`, `applicable_type`, `classes`, … — xuất hiện trong `objects[].tags` của từng file ann, **không** thay thế nhãn phân đoạn pixel. |
  | `projectType` | Ví dụ `"images"`. |
  | `projectSettings` | Cấu hình project (vd. `multiView`). |

  Đếm nhanh: `len(meta["classes"])` = **103** lớp nguyên liệu trong ontology export.

  ---

  ## Chuẩn bị dữ liệu cho train segmentation

  ### Bước bắt buộc

  1. **Ánh xạ tên lớp → id 0..102** (ingredient); **255** hoặc **103** làm **ignore_index** (tuỳ framework); background = một id cố định (thường **103** nếu dùng 0–102 cho ingredient).
  2. **Decode bitmap:** base64 → (thường) **zlib decompress** → PNG → `PIL.Image` → mask nhị phân `H×W`.
  3. **Đặt bitmap vào canvas** đúng `origin` và đúng `(H, W)` của ảnh đang train (sau resize thì **resize nearest-neighbor** cho mask).
  4. **Đã có sẵn pipeline:** đọc trực tiếp **`mask/*.png` + `class_mapping.json`**; hoặc đọc `ann/*.json` (chậm hơn).

  ### Gợi ý tiền xử lý ảnh (giống nhiều paper FoodSeg103)

  - Random resize scale ~0.5–2.0, random crop (vd. 512×512 hoặc 768×768 nếu VRAM đủ), horizontal flip, color jitter.
  - Normalize theo mean/std ImageNet nếu dùng backbone pretrained ImageNet.

  ### Loss / class imbalance

  - Baseline: **CrossEntropy** với `ignore_index` nếu có vùng ignore.
  - Long-tail: **class weights**, **Focal Loss**, hoặc **Lovász-Softmax** (kết hợp CE).

  ### Metrics

  - **mIoU**, **mAcc**, **aAcc** (pixel accuracy) — thống nhất với paper benchmark khi có full split.

  ---

  ## Cấu hình train theo framework

  ### PyTorch “thuần” (Dataset tối thiểu)

  Cần khai báo:

  - `num_classes = 104` (103 + background) **hoặc** 103 nếu không model hóa background riêng (ít gặp).
  - `in_channels = 3`, output logits `B × num_classes × H × W`.
  - Optimizer: **SGD** lr ~0.01, momentum 0.9, weight decay 5e-4 + **poly decay** (power 0.9) là setup gần với nhiều bài segmentation; hoặc **AdamW** lr ~1e-4–6e-4 cho ViT/Swin.
  - Batch: tuỳ GPU; crop lớn thì giảm batch.

  ### MMDetection / MMSegmentation (nếu dùng)

  1. Viết **custom dataset** (hoặc chuyển sang định dạng họ hỗ trợ: mask PNG + `split` list).
  2. Trong config `.py`:
    - `data_root`, `train_pipeline` / `test_pipeline` (Resize, RandomCrop, PhotoMetricDistortion, v.v.).
    - `model.decode_head.num_classes = 104`.
    - `default_hooks.logger` + `CheckpointHook`.
  3. Đăng ký dataset trong registry nếu dùng custom class.

  ### torchvision (DeepLabv3 / FCN)

  - `models.segmentation.deeplabv3_resnet50(weights=..., num_classes=104)`.
  - Dataset trả về `image` float tensor và `mask` `LongTensor` `H×W`.

  ### Hugging Face `transformers` (SegFormer, v.v.)

  - `num_labels=104`, `reduce_labels=False` tùy convention; mask dạng `labels` id từng pixel.

  ---

  ## Kiểm tra nhanh trên máy

  ```powershell
  # PowerShell — đường dẫn gốc project
  (Get-ChildItem source/dataset/foodseg103/train/img).Count
  (Get-ChildItem source/dataset/foodseg103/train/ann).Count
  (Get-ChildItem source/dataset/foodseg103/train/mask).Count
  ```

  Đếm lớp (Python): duyệt mọi `*.json` trong `train/ann` và `test/ann`, `set(o["classTitle"] for o in data["objects"])`.

  ---

  ## Tài liệu / benchmark gốc

  - FoodSeg103: dataset phân đoạn nguyên liệu, 103 lớp, nhãn pixel-level (định dạng gốc có thể là JSON/COCO tùy nguồn tải; bản này là export kiểu **Supervisely bitmap**).

  Khi trích dẫn số ảnh và split, ưu tiên **số liệu đo được từ `foodseg103/` trong repo** hoặc nêu rõ đang dùng **full benchmark** từ tác giả.
