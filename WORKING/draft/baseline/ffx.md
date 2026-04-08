# 1) CONFIG: sửa để phục vụ debug và crop/sampling, không đổi backbone

## Hiện tại cần giữ nguyên

Ở baseline hiện tại, phần model BiSeNet V1 / Xception-like, optimizer SGD, poly LR, checkpoint flow là đủ để tiếp tục debug. Chưa nên đụng vào backbone hay thêm module mới. Điều này cũng đúng với logic debug trước đó: phải chứng minh pipeline học được trước khi thêm texture-aware module.  

## Cần sửa gì trong config

Trong cell config hiện tại, bạn nên thêm hoặc tách rõ các nhóm tham số sau:

### A. Nhóm debug overfit

Thêm:

* `OVERFIT_MODE`
* `OVERFIT_N`
* `DEBUG_DISABLE_HEAVY_AUG`
* `DEBUG_SAVE_VIS_EVERY`

Ý nghĩa:

* bật mode train trên 8 hoặc 32 ảnh
* tắt augmentation mạnh
* dễ debug

### B. Nhóm crop

Thêm:

* `USE_FOREGROUND_CROP = True`
* `CROP_SIZE_BASE = 384 hoặc 448`
* `CROP_SIZE_LARGE = 512 hoặc 640`
* `FOREGROUND_MIN_RATIO = 0.03 ~ 0.08`
* `FOREGROUND_MAX_TRIES = 10`
* `LARGER_CROP_CLASS_IDS = [...]`

Ý nghĩa:

* mặc định crop quanh foreground
* một số class special dùng crop lớn hơn

### C. Nhóm split/sampling

Thêm:

* `USE_WEIGHTED_SAMPLER`
* `USE_HARD_PAIR_SAMPLER`
* `RARE_CLASS_IDS`
* `HARD_PAIR_LIST`
* `SAMPLER_RARE_POWER`
* `SAMPLER_PAIR_BOOST`

### D. Nhóm metric

Thêm:

* `REPORT_FG_PRESENT = True`
* `REPORT_PAIR_CONFUSION = True`
* `REPORT_SMALL_OBJECT_GROUP = True`

## Không nên sửa gì ở config lúc này

* không đổi backbone
* không thêm head mới
* không bật color jitter mạnh lại
* không scale rộng kiểu paper ngay

---

# 2) SPLIT: không chỉ random split, phải thêm audit và cache coverage

## Hiện tại baseline đang làm gì

Baseline đang lấy toàn bộ `train/img` rồi `train_test_split` theo `VAL_RATIO`. Cách này đủ cho smoke test, nhưng chưa đủ cho dataset long-tail như rebalanced FoodSeg. 

## Cần sửa gì

Bạn không nhất thiết đổi split ngay sang multilabel-stratified phức tạp, nhưng phải sửa phần split thành 2 bước:

### Bước 1: split như hiện tại vẫn được

Giữ logic random split để không đổi quá nhiều biến.

### Bước 2: audit coverage ngay sau split

Ngay sau khi tạo `train_samples` và `val_samples`, phải xuất thêm một bảng coverage theo class:

* `train_presence_count`
* `val_presence_count`
* `train_total_pixels`
* `val_total_pixels`

Việc này cần vì rebalanced vẫn còn nhiều class rất hiếm; nếu class rơi lệch split thì model học không nổi dù pipeline đúng. Trong thống kê rebalanced hiện tại vẫn còn nhiều lớp presence thấp và pixel thấp, thậm chí có dead class. 

## Sửa như thế nào

Sau cell split:

* viết hàm quét mask theo danh sách sample
* đếm pixel/presence từng class
* lưu ra CSV

## Sau audit, xử lý ra sao

* nếu class nào `train_presence_count = 0` thì không train tiếp split đó
* nếu nhiều class `train_presence_count <= 2` thì cân nhắc đổi seed hoặc split strategy
* nếu split ổn thì giữ nguyên để giảm số biến thay đổi

## Không nên làm gì

* không vừa đổi split vừa đổi backbone/loss ngay
* không tự assume split hiện tại ổn chỉ vì tổng dataset đã rebalance

---

# 3) TRANSFORM / CROP: đây là phần cần sửa mạnh nhất

Đây là chỗ quan trọng nhất.

## Hiện tại baseline đang làm gì

Train transform hiện tại:

* random scale từ `0.75 → 2.0`
* random horizontal flip
* pad nếu thiếu
* random crop cố định
* normalize 

Logic này gần tinh thần paper BiSeNet, vì paper cũng dùng random scale + crop. Nhưng paper làm trên Cityscapes, còn dataset của bạn có nhiều class hiếm, object nhỏ, cặp texture chồng lấn. BiSeNet paper nhấn mạnh giữ spatial detail và receptive field, nên nếu crop làm mất detail thì tinh thần model cũng bị phá. 

## Cần sửa gì

### A. Tách transform thành 2 mode

Không nên chỉ có 1 `TrainTransform`.

Tách ra:

* `TrainTransformDebug`
* `TrainTransformMain`

#### `TrainTransformDebug`

Dùng cho overfit 8/32:

* resize hoặc pad nhẹ
* flip nhẹ
* foreground-aware crop
* không color jitter
* không random scale rộng

#### `TrainTransformMain`

Dùng cho train chính:

* foreground-aware crop là mặc định
* scale range hẹp hơn
* larger crop cho special classes

---

## B. Thay random crop thuần bằng foreground-aware crop

Notebook phân tích đã có helper `random_foreground_crop`, ưu tiên crop quanh pixel foreground và chỉ nhận crop nếu foreground ratio đủ lớn. 

### Vì sao phải sửa

Dataset hiện tại có:

* foreground ratio không nhỏ, nhưng class nhỏ rất dễ mất trong crop
* nhiều object nhỏ / hard pair / edge vụn
* summary texture đã flag **16 class cần larger_crop** và **53 class cần edge/detail emphasis**. 

### Sửa như thế nào

Trong `__call__` của transform:

* sau resize và flip
* thay `RandomCrop.get_params(...)` bằng foreground-aware crop logic

Cụ thể:

1. lấy mask
2. tìm các pixel foreground
3. chọn tâm crop quanh foreground
4. thử tối đa `max_tries`
5. chỉ nhận crop nếu `(crop_mask != bg).mean() > min_fg_ratio`
6. nếu fail mới fallback random crop

### Không nên làm gì

* không để `min_fg_ratio` quá cao
* không ép crop phải chứa quá nhiều foreground với class nhỏ, vì sẽ làm fail crop liên tục

---

## C. Larger crop cho class mơ hồ / object nhỏ

### Vì sao

Các nhóm như:

* leafy greens
* mushroom-related
* rice vs potato
* pork vs steak
* onion / spring onion
* blueberry / cherry
  cần nhiều context hơn patch/crop thường. Summary texture cũng đã flag nhóm cần larger crop. 

### Sửa như thế nào

Trong transform:

* nếu ảnh chứa class thuộc `LARGER_CROP_CLASS_IDS`
* thì dùng `crop_size_large`
* ngược lại dùng `crop_size_base`

### Cách kiểm tra class trong ảnh

Sau khi load mask:

* `present_classes = np.unique(mask)`
* nếu giao với danh sách special class khác rỗng, bật crop lớn

### Không nên làm gì

* không tăng crop cho toàn bộ dataset ngay
* không vừa larger crop vừa scale xuống quá mạnh

---

## D. Thu hẹp scale augmentation

### Hiện tại

`SCALES = [0.75, 1.0, 1.5, 1.75, 2.0]` trong baseline. 

### Cần sửa

Cho v2 hiện tại, nên giảm thành dải hẹp hơn, ví dụ:

* debug: chỉ `1.0`
* main v2 sớm: `[0.9, 1.0, 1.25, 1.5]`

### Vì sao

Scale rộng làm:

* object nhỏ biến mất
* texture chi tiết bị bể
* hard pair càng khó

---

# 4) DATASET OBJECT: cần trả thêm metadata để phục vụ sampler

## Hiện tại baseline đang trả gì

`FoodSegDataset.__getitem__` đang trả:

* image
* mask
* stem
* img_path
* mask_path 

## Cần sửa gì

Dataset nên trả thêm:

* `present_class_ids`
* `foreground_ratio`
* có thể thêm `contains_hard_pair_class`

### Vì sao

Sampler cần biết:

* ảnh chứa rare class nào
* ảnh chứa hard-pair class nào
* ảnh có foreground ít hay nhiều

### Sửa như thế nào

Trong `__getitem__`:

* sau khi load mask thô hoặc từ transformed mask
* tính:

  * `present = np.unique(mask_np)`
  * bỏ `IGNORE_INDEX`, background
* trả thêm dictionary metadata

Ví dụ:

* `meta = {"present_classes": [...], "fg_ratio": ..., "has_hard_pair": ...}`

### Không nên làm gì

* không nhồi quá nhiều feature phức tạp vào dataset lúc này
* không tính texture descriptor online trong dataset

---

# 5) DATALOADER / SAMPLER: cần thêm sampling có chủ đích

Đây là phần sửa lớn thứ hai sau crop.

## Hiện tại baseline đang làm gì

Train loader đang:

* `shuffle=True`
* không sampler đặc biệt 

## Cần sửa gì

### A. Rare-class sampler

Tạo sample weight theo ảnh.

#### Cách tính

Mỗi ảnh có weight dựa trên các class xuất hiện:

* class càng hiếm trong train thì weight càng cao

Có thể dùng:

* tổng `1 / presence_count[class]`
  hoặc mềm hơn:
* tổng `1 / sqrt(presence_count[class])`

#### Dùng khi nào

Chỉ bật sau khi:

* split đã audit
* crop đã sửa

---

### B. Hard-pair sampler

#### Mục tiêu

Tăng tần suất model nhìn các cặp dễ nhầm.

#### Danh sách pair ban đầu

Lấy đúng từ texture summary:

* chicken duck vs fish
* fish vs pork
* onion vs spring onion
* blueberry vs cherry
* pork vs steak
* rice vs potato 

#### Cách tính weight

Nếu ảnh chứa class thuộc hard-pair list:

* tăng thêm weight

Nếu ảnh chứa cả 2 class trong cùng pair:

* boost mạnh hơn

### Không nên làm gì

* không ép batch hoàn toàn chỉ toàn rare/hard pair
* không boost quá mạnh làm train distribution méo

---

## C. Triển khai trong DataLoader

Khi bật sampler:

* dùng `WeightedRandomSampler`
* tắt `shuffle=True`

---

# 6) LOSS: chỉ sửa sau crop + sampling, nhưng phải chuẩn bị hook từ bây giờ

## Hiện tại baseline đang dùng gì

* `CrossEntropyLoss(ignore_index=255)` cho main + 2 aux heads 

## Cần sửa gì

### Chưa sửa ngay trong run đầu

Run đầu với crop/sampler mới vẫn nên giữ CE thuần để so công bằng.

### Sau đó mới thêm

#### A. Weighted CE

Tính class weight từ train split rebalanced hiện tại:

* không dùng inverse frequency quá gắt
* ưu tiên `median frequency` hoặc `1/sqrt(freq)`

#### B. CE + Dice

Thêm cho foreground nhỏ và boundary classes.

## Không nên làm gì

* không đổi loss đồng thời với đổi crop + sampler trong cùng run
* không dùng focal + dice + weighted ce cùng lúc ngay

---

# 7) METRICS: đây là phần phải sửa ngay, không chờ

## Hiện tại baseline đang log gì

`aAcc`, `mAcc`, `mIoU` từ confusion matrix chung. 

## Cần sửa gì

### A. Tách metric foreground-present

Thêm:

* `mIoU_present`
* `mIoU_fg_present`
* `mAcc_present`
* `mAcc_fg_present`

### Vì sao

Dataset này khó ở foreground fine-grained, không phải background lớn. Đo full mIoU rất dễ bị background che mất tín hiệu.

---

### B. Per-class IoU rõ ràng

Mỗi epoch hoặc mỗi best checkpoint, in:

* IoU từng class xuất hiện trong val

Đặc biệt highlight:

* rare classes
* hard-pair classes
* small-object classes

---

### C. Pair confusion

Phải có thống kê riêng cho các cặp:

* chicken duck ↔ fish
* fish ↔ pork
* onion ↔ spring onion
* blueberry ↔ cherry
* pork ↔ steak
* rice ↔ potato

### Sửa như thế nào

Từ confusion matrix full:

* trích submatrix 2x2 hoặc 3x3 của từng pair
* log riêng

---

### D. Small-object group metric

Tạo group metric cho các class object nhỏ.

### Sửa như thế nào

Từ bảng object size summary hoặc split audit:

* chọn list class nhỏ
* lấy mean IoU riêng của nhóm đó

---

# 8) TRAIN LOOP: cần thêm debug log đúng chỗ, không thêm module

## Hiện tại baseline train loop đã khá ổn

Nó đã có:

* AMP
* poly LR
* debug batch đầu
* validate riêng
* save ckpt 

## Cần sửa gì

### A. Save thêm thông tin experiment

Trong checkpoint hoặc log:

* transform mode
* crop mode
* sampler mode
* rare class list version
* hard pair list version

### B. Validate phải trả thêm metrics mới

Hàm `validate()` hiện chỉ trả `val_loss` + scores chung.
Cần sửa để trả thêm:

* per-class IoU dict
* fg_present scores
* pair confusion summary
* small-object group score

### C. Visualize đúng nhóm khó

Hàm visualize hiện đang lấy batch đầu val rồi show overlay.
Cần sửa để có thêm:

* 1 gallery random
* 1 gallery hard-pair
* 1 gallery small-object

---

# 9) MODEL: chỗ nào chưa sửa

## Giữ nguyên

* `SpatialPath`
* `ContextPath`
* `FFM`
* `SegHead`
* `BiSeNetV1`
  vì hiện tại chưa có đủ bằng chứng là bottleneck nằm ở module. 

## Chưa sửa

* không thêm edge branch
* không thêm texture head
* không đổi backbone
* không đổi sang model khác

---

# 10) Tóm gọn theo cell của baseline hiện tại

Dựa trên file train split-cell hiện tại, đây là các phần nên sửa:

## Sửa mạnh

* **Cell Config**: thêm cờ debug/crop/sampler/metric
* **Cell Split**: thêm audit coverage train/val
* **Cell TrainTransform / ValTransform**: thay random crop bằng foreground-aware crop, thêm larger crop logic
* **Cell Dataset**: trả thêm metadata class/presence
* **Cell DataLoader**: thêm weighted sampler / hard-pair sampler
* **Cell Metrics**: thêm fg-present, per-class, pair confusion, small-object group
* **Cell Validate / Visualize**: log đúng nhóm khó

## Giữ nguyên

* model blocks
* BiSeNet structure
* optimizer SGD
* poly LR
* checkpoint flow

## Chưa động tới

* texture-aware module
* edge branch
* loss phức tạp
* backbone switch

---

# 11) Chốt cuối: cụ thể phải sửa gì trước tiên

Nếu nói cực thực dụng, trên baseline hiện tại bạn phải sửa theo đúng thứ tự code này:

### Sửa ngay

1. **TrainTransform**

   * foreground-aware crop
   * giảm scale range
   * larger crop cho special classes

2. **Split audit**

   * xuất coverage theo class train/val

3. **Dataset + DataLoader**

   * trả metadata ảnh chứa class nào
   * rare sampler
   * hard-pair sampler

4. **Metrics**

   * fg-present metrics
   * per-class IoU
   * pair confusion

### Chưa sửa ngay

5. loss
6. model modules
