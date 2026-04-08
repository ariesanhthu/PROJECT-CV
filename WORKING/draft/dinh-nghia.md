

# 1. Segmentation (Image Segmentation)

## Trích dẫn nguyên văn

> “Image segmentation is the process of partitioning an image into multiple regions or segments in order to simplify or change the representation of an image into something that is more meaningful and easier to analyze.”

(định nghĩa phổ biến trong computer vision literature)

## Viết lại theo văn phong khoa học (tiếng Việt)

Image segmentation là quá trình phân hoạch một ảnh thành nhiều vùng (regions) hoặc segments khác nhau nhằm đơn giản hóa biểu diễn của ảnh, từ đó giúp việc phân tích và hiểu nội dung ảnh trở nên dễ dàng hơn. Mỗi segment thường bao gồm các pixel có đặc trưng tương đồng về màu sắc, cường độ hoặc texture.

---

# 2. Semantic Segmentation

## Trích dẫn nguyên văn từ paper

> “Deep neural networks are very effective in semantic segmentation, that is labeling each region or pixel with a class of objects/non-objects.” 

## Dịch lại theo văn phong khoa học

Semantic segmentation là bài toán gán nhãn cho từng region hoặc từng pixel trong ảnh với một *class* ngữ nghĩa cụ thể (ví dụ: car, road, sky, vegetation). Khác với segmentation thông thường chỉ phân tách các vùng, semantic segmentation thực hiện *pixel-wise classification*, tức là mỗi pixel trong ảnh được gán một nhãn semantic tương ứng với đối tượng hoặc lớp mà nó thuộc về.

---

# 3. Viết ngắn gọn phù hợp cho paper (đề xuất)

Bạn có thể viết gọn trong phần **Introduction** như sau:

```latex
Image segmentation is the process of partitioning an image into multiple regions in order to simplify its representation and facilitate image analysis. 

A more advanced task is semantic segmentation, which performs pixel-wise classification by assigning each pixel in an image to a semantic class. As stated in \cite{lateef2019survey}, semantic segmentation refers to labeling each region or pixel with a class of objects or non-objects.
```

---

# 4. BibTeX reference (LaTeX)

```bibtex
@article{lateef2019survey,
  title={Survey on semantic segmentation using deep learning techniques},
  author={Lateef, Fahad and Ruichek, Yassine},
  journal={Neurocomputing},
  volume={338},
  pages={321--348},
  year={2019},
  publisher={Elsevier}
}
```
