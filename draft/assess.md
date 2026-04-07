## Thông tin chung
Tính toàn vẹn dữ liệu: rất tốt
Test: 2135/2135 ảnh hợp lệ, tỉ lệ ok = 1.000
Train: 3981/3982 ảnh hợp lệ, tỉ lệ ok = 0.999749
Độ phủ nhãn: đầy đủ
Meta có 103 class foreground.
Annotation quan sát được đủ 103/103 class.
=> Không thiếu class nào so với meta.

## Class imbalance
~ 40 Class xuất hiện < 50 lần trong tập dataset. Trong đó, có các class có thể gộp lại như 	king oyster mushroom, enoki mushroom, oyster mushroom, white button mushroom...
![alt text](image.png)
![alt text](image-1.png)

Top-1 class chiếm 5.6% tổng object.
Top-10 class chiếm 44.7% tổng object.
Có 13 class xuất hiện dưới 10 ảnh, đây là phần long-tail cần xử lý khi train.
Median số ảnh trên mỗi class là 81, mức ổn cho phần lớn class.


Chất lượng dataset: tốt, dùng được ngay cho huấn luyện segmentation đa lớp.
Mức đánh giá phù hợp với điểm tự động của notebook: 95/100, mức A.
Rủi ro chính còn lại là long-tail class và chênh lệch scale train-test.
Khuyến nghị trước khi train model:

    - Dùng class-weighted loss hoặc focal loss để giảm bias vào lớp phổ biến.
    - Dùng sampler cân bằng lớp hoặc oversample ảnh chứa lớp hiếm.
    - Theo dõi per-class IoU, không chỉ mIoU tổng.
    - Chuẩn hóa resize/crop đa tỉ lệ để giảm ảnh hưởng domain gap kích thước.