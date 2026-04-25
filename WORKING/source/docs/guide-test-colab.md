Bạn là technical assistant. Hãy giúp tôi tổ chức workflow **test trên Google Colab** sao cho sau này chuyển sang **FPT GPU Cloud** gần như không phải viết lại nhiều.

**Context của tôi**

* Tôi code ở local theo file `.py`
* Tôi push code lên GitHub
* Tôi dùng Colab để test nhanh
* Dataset lấy từ Kaggle
* Sau khi test ổn, tôi sẽ train thật trên FPT GPU Cloud

**Yêu cầu bạn phải trả lời**

1. Hướng dẫn tôi cách tổ chức project để dùng chung cho:

   * local
   * Colab
   * FPT GPU Cloud
2. Đề xuất cấu trúc project:

   * `src/`
   * `configs/`
   * `scripts/`
   * `requirements.txt`
   * `notebooks/`
3. Chỉ rõ phần nào nên nằm trong notebook, phần nào phải chuyển sang file `.py`.
4. Viết cho tôi notebook workflow tối giản:

   * clone GitHub repo
   * cài package
   * tải dataset Kaggle
   * train thử
   * lưu output/checkpoint
5. Chỉ ra các dòng code kiểu Colab-specific cần tránh để sau này dễ chuyển sang FPT:

   * `/content/...`
   * `drive.mount(...)`
   * hardcode path
6. Cho tôi cách chuẩn hóa path/config để sau này chỉ đổi config là chạy được trên cả Colab và FPT.
7. Viết mẫu:

   * `train.py`
   * `config.yaml`
   * cell Colab tối thiểu để chạy train
8. Kết thúc bằng phần: “Muốn chuyển từ Colab sang FPT Cloud thì phải sửa những gì?”

**Output format**

* Trả lời bằng tiếng Việt
* Ngắn gọn nhưng thực chiến
* Ưu tiên code mẫu
* Có checklist migrate từ Colab sang FPT
