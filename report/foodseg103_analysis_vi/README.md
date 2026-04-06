# FoodSeg103 Analysis Report (Tiếng Việt)

## Cấu trúc thư mục
- `main.tex`: file gốc của báo cáo.
- `sections/`: các phần nội dung tách riêng.
- `figures/`: toàn bộ ảnh minh họa dùng trong báo cáo.
- `scripts/generate_figures.py`: script tái tạo ảnh từ CSV + dữ liệu gốc.

## Cách tái tạo ảnh
Chạy từ thư mục gốc dự án:

```powershell
C:/Users/tinal/miniconda3/Scripts/conda.exe run -p C:\Users\tinal\miniconda3 --no-capture-output python d:\CODE\PROJECT-CV\report\foodseg103_analysis_vi\scripts\generate_figures.py
```

## Cách biên dịch PDF
Trong thư mục `report/foodseg103_analysis_vi`:

```powershell
xelatex main.tex
xelatex main.tex
```

Chạy 2 lần để cập nhật mục lục và tham chiếu hình.
