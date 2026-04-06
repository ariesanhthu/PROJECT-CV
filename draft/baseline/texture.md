1. Texture descriptives

class nào mịn, class nào hạt, class nào nhiều cạnh vụn
intra-class variance: cùng một class nhưng texture có ổn định không
inter-class similarity: class nào nhìn gần nhau nhất về texture

2. Texture embedding

lấy patch từ vùng mask của từng class
trích feature từ backbone pretrained hoặc từ filter thủ công
PCA / UMAP / clustering để xem class nào chồng lấn

3. Hard-pair analysis

tìm các cặp class dễ nhầm nhất theo texture
ví dụ rice–mashed potato, pork–steak, mushroom-related, leafy greens...
xem nhầm do texture thật hay do màu/context

4. Texture-aware actionables

class nào cần crop lớn hơn
class nào cần edge/detail branch
class nào nên group sampling theo hard pairs
class nào cần clean annotation trước


---
Ổn rồi. Kết quả này cho thấy pipeline **đã chạy đúng pha 1**: build metadata → extract patch → tính handcrafted texture feature.

Từ log của bạn, có vài ý quan trọng:

### 1) Coverage hiện tại đủ để bắt đầu phân tích, nhưng chưa phải tối ưu

* **3972 samples**
* **7243 patches**
* tức trung bình khoảng **1.82 patch / ảnh**

Cái này đủ để làm:

* texture descriptives
* similarity giữa class
* hard-pair sơ bộ

Nhưng nó vẫn hơi “mỏng” cho các class phổ biến và class có texture đa dạng nội bộ.
Vì hiện tại mỗi class nhiều khả năng chỉ lấy vài component lớn nhất cho mỗi ảnh, nên **intra-class variance** có thể còn bị thiếu.

### 2) Feature table đang đúng hướng

Bạn đã có đủ nhóm feature cần thiết:

* color stats: `r/g/b`, `h/s/v`
* handcrafted texture:

  * `smoothness_score`
  * `granularity_score`
  * `crumbly_edge_score`
* cùng metadata patch:

  * `patch_fg_ratio`
  * `patch_area_pixels`
  * `center_x / center_y`

Tức là bây giờ bạn **đã có nền để làm 4 việc chính**, chưa cần sửa pipeline nữa.

### 3) Nhưng hiện tại bảng này mới là **patch-level**, chưa phải class-level conclusion

Ví dụ 5 dòng đầu chỉ cho thấy:

* `rice` patch có `smoothness_score` khá cao và `granularity_score` cũng cao
* `onion` có `crumbly_edge_score` cao hơn
* `cucumber` có `granularity_score` cao hơn `chicken duck`

Nhưng **không được kết luận class texture từ vài patch đầu**.
Phải gom theo class rồi xem:

* mean
* median
* std
* percentile

Nếu không sẽ lại rơi vào suy diễn.

---

## Giờ nên làm ngay theo thứ tự này

### Bước A — tổng hợp class-level texture descriptives

Tạo bảng theo class:

* `n_patches`
* `smoothness_mean`, `smoothness_std`
* `granularity_mean`, `granularity_std`
* `crumbly_edge_mean`, `crumbly_edge_std`
* `patch_fg_ratio_mean`
* nối thêm:

  * `presence_count`
  * `pixel_ratio`
  * `median_area_ratio`

Mục tiêu:

* class nào thật sự **mịn**
* class nào **hạt**
* class nào **nhiều cạnh vụn**
* class nào **intra-class variance lớn**

### Bước B — lọc class không đủ patch

Đặt ngưỡng:

* `n_patches < 15` hoặc `< 20` → **low confidence**
* không dùng để kết luận mạnh về texture

Cái này rất quan trọng vì dataset của bạn có nhiều class cực hiếm như `kelp`, `enoki mushroom`, `okra`, `peanut`, `egg tart`. 

### Bước C — tách texture khỏi color

Vì feature table của bạn đang có cả màu lẫn texture, cần làm 2 similarity riêng:

* **texture-only similarity**

  * GLCM / LBP / FFT / edge / morphology / 3 score tổng hợp
* **color-only similarity**

  * `r/g/b/h/s/v mean/std`

Sau đó so:

* cặp nào gần nhau vì **texture**
* cặp nào gần nhau vì **màu**
* cặp nào gần nhau vì cả hai

Đây mới là chỗ hard-pair analysis có ý nghĩa.

### Bước D — nối với statistics

Với mỗi class, ghép thêm:

* `presence_count`
* `pixel_ratio`
* `median_area_ratio`

Vì actionables phải dựa trên cả:

* texture khó
* rarity
* object size nhỏ

Chứ không chỉ texture đơn lẻ.
Điều này khớp với statistics full của bạn: nhiều class vừa hiếm vừa nhỏ.  

---

## Cách đọc actionables sau này

Bạn nên mong script cuối cùng trả ra kiểu này:

### `need_larger_crop`

Ưu tiên nếu:

* `median_area_ratio` nhỏ
* `patch_fg_ratio_mean` thấp
* texture quan trọng nhưng object nhỏ

Nhóm kiểu:

* peanut
* dried cranberries
* blueberry
* spring onion
* garlic
  rất dễ rơi vào đây. 

### `need_edge_detail_branch`

Ưu tiên nếu:

* `crumbly_edge_mean` cao
* `edge_density` cao
* `fragmentation` cao

Nhóm rau lá, topping vụn, thành phần mép răng cưa sẽ đáng nghi hơn.

### `need_group_sampling`

Ưu tiên nếu:

* `presence_count` rất thấp
* hoặc có hard texture partners rõ ràng

### `need_annotation_audit`

Ưu tiên nếu:

* variance nội lớp rất lớn bất thường
* nhưng sample count lại thấp
* hoặc patch của cùng class tách thành nhiều cluster rời nhau

---

## Điều tôi thấy hợp lý nhất từ run này

Run của bạn đã chứng minh:

* pipeline không bị lỗi logic cơ bản
* patch extraction đang chạy được trên full train
* feature table đủ giàu để phân tích texture có phương pháp

Nói cách khác:

> **Bạn đã xong bước “thu thập bằng chứng texture”. Giờ không nên sửa extractor nữa, mà phải chuyển sang bước “tổng hợp class-level + similarity + actionables”.**

---

## Việc nên làm tiếp ngay

1. Tạo `texture_descriptives_by_class.csv`
2. Tạo `texture_similarity_matrix.csv`
3. Tạo `texture_vs_color_pairs.csv`
4. Tạo `texture_actionables.csv`

