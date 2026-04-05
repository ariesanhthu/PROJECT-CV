Quyết định cấu hình cuối cùng

Với đúng dataset copy này, mình chốt cấu hình phù hợp là:

BiSeNetV2
104 classes
crop 512
SGD lr 0.01
80 epochs
CrossEntropy có class weights
val = 10% từ train
không ignore background, vì background đang là class thật 103

Lý do chính là bộ train hiện chỉ có 922 ảnh và long-tail rất mạnh, nên 512 + weighted CE là điểm khởi đầu an toàn hơn so với crop lớn hoặc loss quá phức tạp.