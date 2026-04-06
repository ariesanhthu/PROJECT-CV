> **xác định chính xác vì sao model không học được**.

Và với log bạn đã chạy, dấu hiệu là model đang **học rất yếu**: sau nhiều epoch, `mIoU` chỉ quanh vài phần trăm, `aAcc` lại quanh ~0.48–0.52, kiểu này thường là model đang học nền/lớp lớn nhiều hơn là học phân đoạn thực sự. 

## Mình chốt thứ tự ưu tiên như này

### Giai đoạn 1 — đừng đổi backbone, đừng thêm module

Trước hết, **giữ nguyên model** và chỉ làm chẩn đoán.

Lý do:

* nếu vừa đổi backbone, vừa đổi split, vừa đổi loss, vừa clean dataset, bạn sẽ **không biết cái nào gây ra cải thiện hoặc làm tệ đi**;
* hiện tại vấn đề chưa chắc là do backbone, mà rất có thể do **split + loss + sampling + taxonomy/data consistency**.

---

# 1) Việc số 1: làm một bài test “model có khả năng overfit tập cực nhỏ không?”

Đây là phép thử quan trọng nhất.

## Cách làm

Lấy:

* 8 ảnh
* rồi 32 ảnh

Tắt gần hết augmentation mạnh:

* bỏ random scale quá lớn
* bỏ color jitter
* giữ resize/crop đơn giản

Train chỉ trên đúng tập nhỏ đó và eval luôn trên chính nó.

## Kỳ vọng

Nếu pipeline đúng, model phải:

* overfit được rất nhanh
* loss tụt mạnh
* mIoU trên chính tập nhỏ phải lên rất cao

## Nếu **không overfit nổi 8–32 ảnh**

thì gần như chắc chắn bạn đang có lỗi ở một trong các nhóm này:

* mapping / mask / nhãn
* loss / output channel / class indexing
* crop làm mất object
* learning rate / optimization
* metric hoặc validation logic

Đây là bước quan trọng nhất vì nó tách bạch:

* **“model yếu vì bài toán khó”**
  với
* **“pipeline đang có bug hoặc thiết kế sai”**

---

# 2) Việc số 2: xác minh dataset-consistency trước khi train lại dài

Hiện đã có một tín hiệu đỏ rõ ràng:

* có class `0 pixel`, `0 presence_count` trong dataset rebalanced hiện tại. 

Ngoài ra dataset của bạn imbalance rất mạnh, nhiều class cực hiếm, nhiều object nhỏ. 

## Cần check ngay 4 thứ

### A. Có class nào trong ontology nhưng không có pixel?

Class kiểu đó là dead class.

### B. Có class nào trong **train split** bị mất hoàn toàn không?

Không chỉ toàn dataset, mà riêng **train** mới là thứ quyết định model có học được không.

### C. Mask có đúng dải nhãn `0..76` không?

Tài liệu dataset rebalanced của bạn ghi rõ:

* background `0`
* foreground `1..76`
* tổng `77` classes. 

### D. Một vài ảnh/mask có đúng nghĩa không?

Mở ngẫu nhiên:

* ảnh
* mask
* overlay
* class names present
  để chắc là mapping và rasterization đúng.

## Nếu bước này chưa sạch

thì **đừng train dài tiếp**.

---

# 3) Việc số 3: sửa split trước, nhưng chỉ sửa split

Hiện split của bạn là:

* random seed 42
* lấy 10% train làm val
* lưu JSON cố định. 

Cách này tốt ở chỗ reproducible, nhưng với dataset long-tail thì có rủi ro:

* class hiếm bị rơi khỏi train
* hoặc train còn quá ít ảnh của class hiếm

## Việc nên làm

Không cần đổi ngay sang thứ quá phức tạp, nhưng hãy làm:

### Check coverage hiện tại

In ra cho từng class:

* `train_presence_count`
* `val_presence_count`

Tìm các class có:

* `train_presence_count = 0`
* hoặc `train_presence_count খুব thấp`

## Sau đó mới quyết định

Nếu có nhiều class bị mất khỏi train:

* đổi sang **presence-aware / multilabel-stratified split**

Nếu coverage train ổn:

* có thể giữ split cũ tạm thời để giảm số biến thay đổi

=> Nghĩa là: **không phải cứ nghe “multilabel-stratified” là làm ngay**, mà phải dùng nó để giải quyết một lỗi đã xác nhận.

---

# 4) Việc số 4: kiểm tra crop/augmentation có đang phá bài toán không

Hiện bản cũ của bạn dùng:

* random scale khá rộng
* random crop 384
* color jitter. 

Trong khi phân tích dataset cho thấy:

* nhiều class rất hiếm
* nhiều object rất nhỏ
* nhiều class có median area ratio nhỏ. 

Điều này rất nguy hiểm:

* random crop dễ cắt mất object hiếm
* scale xuống nhiều làm object nhỏ gần như biến mất
* augmentation mạnh làm nhiễu hơn là giúp

## Test nên làm

Tạo 2 run rất nhỏ:

### Run A

augmentation hiện tại

### Run B

augmentation tối giản:

* resize/pad
* horizontal flip
* normalize
* không color jitter
* crop nhẹ hoặc foreground-aware crop

Nếu B học rõ ràng hơn A, nghĩa là augmentation/crop đang phá signal.

---

# 5) Việc số 5: giữ backbone, chỉ thử loss/sampling sau khi qua 4 bước trên

Đây là chỗ nhiều người nhảy vào quá sớm.

Hiện dataset của bạn:

* nền lớn
* class imbalance mạnh
* nhiều class ít ảnh. 

Nên loss CE thuần có thể chưa đủ tốt. Nhưng:

**đừng đổi loss trước khi chứng minh pipeline sạch**.

## Sau khi đã qua bài test overfit + split + dataset check

thì thứ nên thử đầu tiên là:

* **Weighted CE + Dice**
  hoặc
* giữ CE nhưng thêm **foreground-aware crop / sampler**

Chưa cần:

* đổi backbone
* thêm texture modules
* thêm attention mới

---

# 6) Backbone hiện tại chưa phải nghi phạm số 1

Code của bạn là custom BiSeNet-style với ResNet18 pretrained, không phải exact paper BiSeNet, nhưng nó vẫn đúng tinh thần SP/CP/ARM/FFM.

Nên ở thời điểm này:

* **đừng đổ lỗi cho backbone trước**
* vì một model khá ổn như vậy mà vẫn không overfit nổi tập nhỏ, thì gần như lỗi nằm ở pipeline/dataset chứ không phải kiến trúc

---

# 7) Thứ tự mình khuyên bạn làm, rất cụ thể

## Bước 1

**Overfit 8 ảnh**

* nếu fail → dừng toàn bộ tối ưu khác, debug pipeline

## Bước 2

**Audit train split coverage**

* class nào mất khỏi train?
* class nào chỉ còn 1–2 ảnh?

## Bước 3

**Audit taxonomy**

* class 0-pixel
* dead classes
* mapping consistency

## Bước 4

**So sánh augmentation mạnh vs augmentation tối giản**

* để xem crop/scale có phá object nhỏ không

## Bước 5

**Chỉ khi 4 bước trên ổn mới thử**

* weighted CE + Dice
* foreground-aware crop
* weighted sampler

## Bước 6

**Sau cùng mới nghĩ tới**

* đổi backbone
* thêm texture block
* thêm edge head
* t-SNE nâng cao
* clean phức tạp hơn

---

# 8) Mục tiêu chẩn đoán của bạn nên là gì?

Đừng hỏi:

> “làm sao để tăng mIoU ngay?”

Hãy hỏi theo thứ tự:

### Câu 1

**Model có overfit nổi tập nhỏ không?**

### Câu 2

**Train split có làm mất class không?**

### Câu 3

**Mask / mapping / num_classes có nhất quán không?**

### Câu 4

**Crop và augmentation có đang phá sample nhỏ/hiếm không?**

### Câu 5

**Sau khi 4 cái trên ổn, loss/sampling nào giúp hơn?**

Chỉ cần trả lời được 5 câu này, bạn sẽ biết vì sao nó không học.

---

# 9) Chốt ngắn gọn

Lúc này **đừng đổi backbone trước**.
Cũng **đừng clean toàn dataset hàng loạt trước**.
Cũng **đừng thêm cả đống augmentation/sampler/module mới ngay**.

Thứ bạn cần làm ngay là:

> **thiết kế một quy trình debug tối thiểu để chứng minh pipeline có học được hay không.**

Nếu phải chọn **1 việc duy nhất ngay bây giờ**, mình chọn:

**train overfit trên 8 ảnh và 32 ảnh, với augmentation tối giản**.

Bước đó sẽ nói cho bạn biết rất nhiều:

* nếu fail → pipeline/data có vấn đề
* nếu pass → model học được, lúc đó mới quay sang split/loss/imbalance

Mình có thể viết ngay cho bạn một checklist debug 1 buổi gồm **5 cell test tuần tự** để xác định nguyên nhân nó không học.
