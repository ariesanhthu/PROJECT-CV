## Fact bắt buộc phải dùng

- Dataset rebalanced có `75` class foreground, không phải `76`.
- Tổng dữ liệu: `6116` ảnh = `3981 train` + `2135 test`.
- Tổng object: `22650`.
- Không có dead class, không có class `0 object`.
- Audit remap sạch: không có lỗi mapping class do rebalance.
- Ảnh không hư, JSON không hư, bitmap decode được toàn bộ.
- Có đúng `4` anomaly đã biết:
  - `train/img/00000273.jpg`: JSON size bị đảo `(512, 384)` vs ảnh thật `(384, 512)`
  - `train/img/00002585.jpg`: JSON size bị đảo `(512, 384)` vs ảnh thật `(384, 512)`
  - `train/img/00003969.jpg`: JSON size bị đảo `(512, 384)` vs ảnh thật `(384, 512)`
  - `train/00003969.jpg`: có `1` mask `pork` out-of-bounds
- Long-tail vẫn nặng. Class cực hiếm gồm:
  - `hamburg(6)`, `seaweed(7)`, `apricot(10)`, `fig(10)`, `crab(13)`, `watermelon(16)`, `wonton dumplings(16)`, `eggplant(17)`, `dried cranberries(18)`, `salad(20)`
- Class chiếm pixel lớn:
  - `chicken duck`, `bread`, `potato`, `pork`, `steak`, `rice`

## Mục tiêu

Đừng tối ưu chung chung. Hãy sửa để pipeline:

1. train đúng `num_classes`
2. fail sớm nếu label/config sai
3. không để augmentation phá object nhỏ và class hiếm
4. có debug path ngắn để chứng minh model học được

## Việc phải làm

### 1. Sửa assumption sai về dataset

- Đồng bộ mọi chỗ đang giả định sai số class.
- Nếu code đang dùng `76` foreground hoặc `77` total classes thì sửa lại theo dataset rebalanced hiện tại.
- Thêm assert/log để fail sớm nếu `num_classes` trong model, loss, decoder, metrics, dataset loader lệch nhau.

### 2. Chặn 4 anomaly đã biết

- Trong dataset loader hoặc preprocess, thêm check để:
  - phát hiện annotation `size` bị đảo
  - tự sửa nếu sửa an toàn
  - hoặc skip có log rõ ràng nếu không sửa an toàn
- Với mask `pork` out-of-bounds của `00003969.jpg`, xử lý dứt điểm:
  - clip hợp lệ nếu pipeline rasterization cho phép
  - nếu không thì skip object/sample có log
- Không được im lặng nuốt lỗi.

### 3. Audit split coverage

- In `train_presence_count` và `val/test_presence_count` theo class.
- Chỉ ra class nào biến mất khỏi train hoặc quá ít mẫu trong train.
- Nếu split hiện tại làm mất class hiếm khỏi train thì đề xuất hoặc triển khai split presence-aware.
- Nếu coverage train ổn thì giữ split cũ.

### 4. Giảm augmentation phá tín hiệu

- Tạo một config augmentation tối giản cho debug:
  - resize/pad ổn định
  - horizontal flip
  - normalize
  - bỏ color jitter mạnh
  - hạn chế random scale quá rộng
  - tránh crop làm mất object nhỏ
- Nếu đang crop, ưu tiên `foreground-aware crop` hoặc crop nhẹ hơn.
- Giải thích ngắn vì sao thay đổi này hợp với long-tail + small objects.

### 5. Viết debug path ngắn, chạy được ngay

- Tạo quy trình debug tối thiểu:
  - overfit `8` ảnh
  - nếu pass thì overfit `32` ảnh
  - sau đó mới train bình thường
- Log đủ để trả lời 4 câu:
  - data vào có đúng label range không
  - sample có class nào không xuất hiện không
  - model có overfit được tập nhỏ không
  - augmentation hiện tại có phá học không

### 6. Chỉ tối ưu loss/sampler sau khi pipeline sạch

- Sau khi 5 bước trên ổn mới thử:
  - `Weighted CE + Dice`
  - weighted sampler
  - foreground-aware crop
- Không đổi backbone trước.
- Không thêm module mới trước.

## Output tôi muốn

- Chỉ sửa những gì cần để pipeline sạch và dễ debug hơn.
- Trả về:
  - các file đã sửa
  - assumption sai nào đã được fix
  - anomaly handling đã thêm ở đâu
  - config augmentation debug mới
  - checklist chạy thử ngắn

## Cấm

- Không được nói lại giả định sai kiểu `77 classes`, `dead class`, `class 0-pixel`.
- Không được đề xuất đổi backbone trước khi chứng minh pipeline hiện tại fail vì model chứ không phải data/pipeline.
- Không được trả lời high-level; phải sửa cụ thể vào code/config nếu tìm thấy chỗ cần sửa.
