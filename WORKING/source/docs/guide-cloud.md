Bạn là technical assistant. Hãy hướng dẫn tôi setup **train thật trên FPT GPU Cloud** theo workflow thực dụng, ít lỗi, ưu tiên giữ được checkpoint/model lâu dài.

**Context của tôi**

* Tôi code ở local, quản lý code bằng GitHub
* Tôi muốn train thật trên FPT GPU Cloud bằng **GPU Container**
* Tôi muốn lưu **checkpoint / final model / logs / dataset cache**
* Tôi muốn có thể SSH từ máy local vào cloud container
* Tôi muốn dùng môi trường ổn định để train lâu, không phải setup lại nhiều lần

**Yêu cầu bạn phải trả lời**

1. Chọn giúp tôi **template phù hợp nhất** giữa Jupyter / PyTorch / Ubuntu / Code Server, và giải thích ngắn gọn vì sao.
2. Hướng dẫn **step-by-step** từ lúc tạo container:

   * chọn template
   * chọn GPU
   * thêm **SSH key**
   * thêm **Persistent Disk**
   * thêm env vars nếu cần
   * cách lấy **SSH command**
3. Đề xuất cấu trúc thư mục chuẩn trên remote:

   * code
   * data
   * checkpoints
   * models
   * logs
4. Chỉ rõ cái gì nên lưu ở **Persistent Disk**, cái gì không nên để ở Temporary Disk.
5. Hướng dẫn workflow chuẩn:

   * code local
   * push GitHub
   * SSH vào container
   * clone repo
   * cài requirements
   * chạy train
   * resume từ checkpoint
6. Viết cho tôi:

   * 1 file `requirements.txt` mẫu
   * 1 file `train.py` mẫu có save checkpoint
   * 1 lệnh train mẫu
   * 1 lệnh resume mẫu
7. Thêm best practices:

   * tránh mất dữ liệu
   * tránh bị tính phí oan khi stop container
   * cách backup model/checkpoint định kỳ
8. Nếu có chỗ nào dễ sai với FPT Cloud thì cảnh báo rõ.

**Output format**

* Trả lời bằng tiếng Việt
* Ưu tiên cực kỳ thực dụng
* Có checklist từng bước
* Có code mẫu ngắn
* Không giải thích lan man
