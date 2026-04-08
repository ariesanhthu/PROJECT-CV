Có. Nếu phân tích theo ngôn ngữ **toán – tin học**, thì “trích xuất texture” có thể hiểu là:

> xây một ánh xạ
> [
> \phi_{\text{tex}}: \mathbb{R}^{H\times W\times C}\rightarrow \mathbb{R}^{h\times w\times d}
> ]
> sao cho đặc trưng đầu ra giữ được các **mẫu biến thiên cục bộ lặp lại** của ảnh: tần số, hướng, độ thô–mịn, tương quan không gian, vi cấu trúc bề mặt.

Trong bài toán của bạn, texture đặc biệt quan trọng vì FoodSeg103 là bài toán **fine-grained ingredient segmentation**: nhiều lớp khác nhau không phân biệt tốt bằng hình dạng tổng thể mà bằng tín hiệu cục bộ như texture, màu và trạng thái bề mặt.  Đề cương của bạn cũng đặt đúng vấn đề này: mô hình lightweight dễ mất thông tin texture khi downsample sớm. 

Mình sẽ tách phân tích thành 5 lớp: **định nghĩa toán học, đặc trưng cổ điển, CNN, frequency, và ứng dụng cho segmentation**.

---

# 1) Texture là gì theo góc nhìn toán học?

Giả sử ảnh xám là hàm:
[
I:\Omega\subset\mathbb{Z}^2\to \mathbb{R}
]
với (I(x,y)) là cường độ tại pixel ((x,y)).

Khác với **shape**, texture không chủ yếu nằm ở biên đối tượng toàn cục mà ở **thống kê cục bộ** của tín hiệu trong một lân cận:
[
\mathcal{N}_{r}(x,y)={(u,v): |(u,v)-(x,y)|\le r}
]

Một cách phát biểu chuẩn là:

* **shape** phụ thuộc mạnh vào hình học biên mức lớn,
* **texture** phụ thuộc vào phân bố cục bộ của:

  * gradient,
  * tần số không gian,
  * hướng,
  * tương quan giữa các pixel,
  * sự lặp lại vi mẫu.

Nói cách khác, texture là một thuộc tính của **phân phối cục bộ**:
[
T(x,y)=\Psi\big(I|_{\mathcal{N}_r(x,y)}\big)
]
trong đó (\Psi) là một toán tử rút trích thống kê.

---

# 2) Các cách trích xuất texture cổ điển

## 2.1. Dựa trên gradient và đạo hàm

Texture thường gắn với biến thiên cục bộ của ảnh. Ta lấy gradient:
[
\nabla I(x,y)=\left(\frac{\partial I}{\partial x},\frac{\partial I}{\partial y}\right)
]
và độ lớn:
[
|\nabla I(x,y)|=\sqrt{I_x^2+I_y^2}
]

Ý nghĩa:

* vùng phẳng: gradient nhỏ
* vùng có texture mạnh: gradient biến đổi dày đặc, đa hướng
* vùng biên rõ: gradient mạnh nhưng tập trung theo đường biên

Để phân biệt texture với biên đơn lẻ, người ta xét **thống kê gradient trên cửa sổ**:
[
\mu_g(x,y)=\frac{1}{|\mathcal N|}\sum_{(u,v)\in\mathcal N}|\nabla I(u,v)|
]
[
\sigma_g^2(x,y)=\frac{1}{|\mathcal N|}\sum_{(u,v)\in\mathcal N}\big(|\nabla I(u,v)|-\mu_g\big)^2
]

Ngoài ra có thể dùng Hessian:
[
H_I=
\begin{bmatrix}
I_{xx} & I_{xy}\
I_{yx} & I_{yy}
\end{bmatrix}
]
để bắt ridge, blob, cấu trúc vi mô.

### Góc nhìn tin học

Đây là một phép **feature engineering cục bộ** từ ảnh đầu vào, độ phức tạp tuyến tính theo số pixel:
[
O(HW)
]

---

## 2.2. Local Binary Pattern (LBP)

LBP là một đặc trưng texture kinh điển.

Với tâm (g_c) và (P) điểm lân cận (g_p), ta định nghĩa:
[
\mathrm{LBP}*{P,R}=\sum*{p=0}^{P-1}s(g_p-g_c),2^p
]
với
[
s(z)=
\begin{cases}
1 & z\ge 0\
0 & z<0
\end{cases}
]

Ý nghĩa:

* mã hóa **quan hệ thứ tự cục bộ** giữa pixel trung tâm và các pixel xung quanh,
* khá bền với thay đổi sáng tuyến tính,
* nhạy với micro-pattern như:

  * cạnh,
  * góc,
  * đốm,
  * vùng hạt.

Ta thường lấy histogram của LBP trong một vùng:
[
h_k = \sum_{(x,y)\in \mathcal R}\mathbf 1{\mathrm{LBP}(x,y)=k}
]
Histogram này chính là vector texture của vùng đó.

### Diễn giải tin học

LBP biến một patch ảnh thành một vector đếm mẫu nhị phân.
Đây là kiểu:

* **quantization cục bộ**
* **histogram-based representation**

Nó bỏ qua hình dạng toàn cục nhưng giữ rất tốt mẫu cục bộ lặp.

---

## 2.3. Gray-Level Co-occurrence Matrix (GLCM)

GLCM mô tả thống kê đồng xuất hiện của mức xám.

Với offset (\delta=(\Delta x,\Delta y)), định nghĩa:
[
P_\delta(i,j)=
#{(x,y): I(x,y)=i,\ I(x+\Delta x,y+\Delta y)=j}
]

Sau chuẩn hóa:
[
\hat P_\delta(i,j)=\frac{P_\delta(i,j)}{\sum_{i,j}P_\delta(i,j)}
]

Từ đó ta tính các texture measures:

**Contrast**
[
\mathrm{Contrast}=\sum_{i,j}(i-j)^2\hat P(i,j)
]

**Energy**
[
\mathrm{Energy}=\sum_{i,j}\hat P(i,j)^2
]

**Homogeneity**
[
\mathrm{Homogeneity}=\sum_{i,j}\frac{\hat P(i,j)}{1+|i-j|}
]

**Entropy**
[
\mathrm{Entropy}=-\sum_{i,j}\hat P(i,j)\log \hat P(i,j)
]

Ý nghĩa:

* texture thô, nhám, lộn xộn → contrast/entropy cao
* texture đều, mịn → homogeneity/energy cao

### Góc nhìn toán học

GLCM chính là một xấp xỉ của **phân phối xác suất cặp pixel có điều kiện theo khoảng cách/hướng**.

### Góc nhìn tin học

Đây là mô hình hóa texture bằng **thống kê bậc hai** của ảnh.

---

## 2.4. Gabor filters

Gabor rất quan trọng vì texture thường mang tính **tần số + hướng**.

Một kernel Gabor 2D:
[
g(x,y)=\exp\left(-\frac{x'^2+\gamma^2 y'^2}{2\sigma^2}\right)\cos\left(2\pi \frac{x'}{\lambda}+\psi\right)
]
trong đó
[
x'=x\cos\theta+y\sin\theta,\qquad
y'=-x\sin\theta+y\cos\theta
]

Tham số:

* (\theta): hướng
* (\lambda): bước sóng / tần số
* (\sigma): độ rộng Gaussian
* (\gamma): độ dẹt

Đặc trưng texture:
[
F_{\theta,\lambda}(x,y)=(I*g_{\theta,\lambda})(x,y)
]

Nếu dùng bank nhiều hướng, nhiều tần số:
[
\Phi_{\text{Gabor}}(x,y)={F_{\theta_k,\lambda_m}(x,y)}_{k,m}
]

### Ý nghĩa

Gabor hoạt động như bộ dò:

* vân ngang/dọc/chéo,
* hạt nhỏ/lớn,
* cấu trúc tuần hoàn.

### Liên hệ CNN

Conv kernel đầu của CNN thường học ra các bộ lọc rất giống Gabor.

---

## 2.5. Fourier / Wavelet / Frequency-domain

Texture hay xuất hiện như mẫu tần số không gian.

Với Fourier transform:
[
\hat I(\omega_x,\omega_y)=\sum_{x,y}I(x,y)e^{-j(\omega_x x+\omega_y y)}
]

Phổ biên độ:
[
A(\omega_x,\omega_y)=|\hat I(\omega_x,\omega_y)|
]

Ý nghĩa:

* texture mịn → năng lượng ở tần số cao
* texture thô → năng lượng ở dải tần thấp hơn
* texture có hướng → phổ lệch theo hướng nhất định

Ta có thể dùng năng lượng theo vành tần số:
[
E(r)=\sum_{\sqrt{\omega_x^2+\omega_y^2}\in [r,r+\Delta r]} A(\omega_x,\omega_y)^2
]

Hoặc theo hướng:
[
E(\theta)=\sum_{(\omega_x,\omega_y)\in \Theta} A(\omega_x,\omega_y)^2
]

### Wavelet

Wavelet tốt hơn Fourier khi cần **vừa cục bộ không gian vừa cục bộ tần số**:
[
W_\psi(a,b)=\int I(t)\psi_{a,b}(t),dt
]
Trong ảnh 2D, wavelet tách texture theo:

* scale,
* vị trí,
* hướng.

---

# 3) Trích xuất texture trong CNN theo ngôn ngữ toán – tin học

Trong CNN, mỗi layer là ánh xạ:
[
X^{(l+1)}=\sigma\big(W^{(l)} * X^{(l)} + b^{(l)}\big)
]

Texture xuất hiện mạnh nhất ở **các tầng nông** vì:

* receptive field nhỏ,
* đặc trưng còn gần với tín hiệu ảnh gốc,
* kernel học cạnh, sọc, góc, hạt.

## 3.1. Vì sao conv trích được texture?

Một kernel (K\in\mathbb{R}^{k\times k}) là một bộ dò mẫu cục bộ.
Đầu ra tại vị trí ((x,y)):
[
Y(x,y)=\sum_{i,j}K(i,j),I(x+i,y+j)
]

Nếu (K) “khớp” với một micro-pattern, đáp ứng sẽ lớn.
Vì thế, một bank conv filters học ra một tập detector:
[
{K_1,\dots,K_m}
]
và tạo tensor feature:
[
Y_c = K_c * I
]

Khi texture khác nhau có các mẫu cục bộ khác nhau, không gian đặc trưng này sẽ tách chúng ra.

---

## 3.2. Texture = thống kê của feature maps

Không chỉ giá trị từng pixel feature, mà cả **thống kê tương quan giữa kênh** cũng biểu diễn texture.

Ví dụ Gram matrix:
[
G_{ij}=\sum_{p}F_{ip}F_{jp}
]
với (F\in \mathbb{R}^{C\times N}), (N=H\times W).

Gram matrix đo mức đồng hoạt hóa giữa các filter, rất nổi tiếng trong style transfer, và “style” ở đây gần với texture.

### Diễn giải

Nếu hai vùng ảnh có:

* shape khác nhau
* nhưng thống kê Gram gần nhau

thì chúng có thể chia sẻ texture tương tự.

---

## 3.3. Tại sao downsampling sớm làm mất texture?

Giả sử downsample bởi stride 2:
[
I'(x,y)=I(2x,2y)
]

Theo Nyquist, thành phần tần số cao nếu không được low-pass thích hợp sẽ:

* bị aliasing,
* hoặc bị loại bỏ.

Texture mảnh, hạt nhỏ, biên yếu thường nằm ở dải cao tần.
Vì thế mô hình lightweight hay aggressive downsampling sẽ mất:
[
\text{high-frequency local cues}
]

Điều này rất đúng với nhận định trong đề cương của bạn: backbone nhẹ hoặc giảm độ phân giải sớm có thể làm suy giảm biểu diễn đặc trưng cục bộ như texture. 

---

# 4) Texture-aware feature extraction cho segmentation

Bây giờ chuyển từ “trích texture” sang “dùng texture trong semantic segmentation”.

Ta cần học một hàm phân đoạn:
[
f_\theta: \mathbb{R}^{H\times W\times C}\to {1,\dots,K}^{H\times W}
]

Nếu muốn texture-aware, ta không chỉ dựa vào semantic feature (F_{\text{sem}}), mà thêm một nhánh texture:
[
F_{\text{tex}} = \phi_{\text{tex}}(I)
]
[
F_{\text{fuse}} = \mathcal{G}(F_{\text{sem}},F_{\text{tex}})
]
[
\hat Y = \mathrm{Decoder}(F_{\text{fuse}})
]

Trong đó:

* (\phi_{\text{tex}}): module texture
* (\mathcal{G}): phép hợp nhất

## 4.1. Các kiểu (\phi_{\text{tex}})

### Cách 1: Hand-crafted texture branch

[
F_{\text{tex}} = \mathrm{Conv}\big(\mathrm{LBP}(I)\big)
]
hoặc
[
F_{\text{tex}} = \mathrm{Conv}\big(\mathrm{GaborBank}(I)\big)
]

Ưu:

* diễn giải được
* phù hợp nghiên cứu nhỏ

Nhược:

* khó tối ưu end-to-end hoàn toàn

---

### Cách 2: Shallow CNN branch

Dùng nhánh nông, receptive field nhỏ:
[
F_{\text{tex}}^{(1)}=\sigma(W_1*I)
]
[
F_{\text{tex}}^{(2)}=\sigma(W_2*F_{\text{tex}}^{(1)})
]

Có thể tránh pooling mạnh để giữ high-frequency detail.

Ưu:

* end-to-end
* gần với BiSeNet Spatial Path

Nhược:

* chưa chắc “texture-aware” thật nếu không có bias kiến trúc phù hợp

---

### Cách 3: Frequency branch

Tách ảnh thành low/high frequency:
[
I = I_{\text{low}} + I_{\text{high}}
]
với
[
I_{\text{low}} = G_\sigma * I,\qquad
I_{\text{high}} = I - I_{\text{low}}
]

Rồi trích đặc trưng:
[
F_{\text{tex}}=\phi(I_{\text{high}})
]

Ưu:

* nhắm trực tiếp vào thành phần texture
* hợp lý cho food segmentation

---

### Cách 4: Multi-scale local statistics

Xét nhiều cửa sổ:
[
T_s(x,y)=\Psi\big(I|*{\mathcal N_s(x,y)}\big),\quad s\in{3,5,7,\dots}
]
rồi ghép:
[
F*{\text{tex}} = [T_{s_1},T_{s_2},\dots,T_{s_n}]
]

Ý nghĩa:

* texture nhỏ và texture lớn được mô tả đồng thời.

---

# 5) Texture-aware trong BiSeNet nên hiểu thế nào?

BiSeNet gốc có:

* **Spatial Path**: giữ spatial detail
* **Context Path**: giữ semantic context
  nhưng **không có module texture-aware tường minh**. 

Theo ngôn ngữ toán – tin học, Spatial Path của BiSeNet gần với:
[
F_{\text{sp}}=\phi_{\text{sp}}(I)
]
trong đó (\phi_{\text{sp}}) cố giữ độ phân giải cao hơn để hạn chế mất chi tiết.

Nhưng:

* nó không tối ưu trực tiếp cho texture statistics,
* không tách explicit high-frequency signal,
* không có texture prior / texture loss riêng.

Vì vậy, nếu muốn “texture-aware BiSeNet”, bạn cần biến:
[
F_{\text{fuse}}=\mathcal G(F_{\text{sp}},F_{\text{ctx}})
]
thành
[
F_{\text{fuse}}=\mathcal G(F_{\text{sp}},F_{\text{ctx}},F_{\text{tex}})
]

---

# 6) Một số công thức hợp nhất texture vào mô hình

## 6.1. Concatenation

[
F_{\text{fuse}} = \mathrm{Conv}\big([F_{\text{sem}};F_{\text{tex}}]\big)
]

Đơn giản, hiệu quả, dễ cài.

---

## 6.2. Attention-guided fusion

Tạo trọng số theo texture:
[
A=\sigma(\mathrm{MLP}(\mathrm{GAP}(F_{\text{tex}})))
]
[
F'*{\text{sem}} = A\odot F*{\text{sem}}
]

Ý nghĩa:

* texture branch đóng vai trò “hướng dẫn”
* kênh semantic nào hữu ích cho texture sẽ được nhấn mạnh

---

## 6.3. Residual enhancement

[
F_{\text{out}} = F_{\text{sem}} + \lambda F_{\text{tex}}
]

Đơn giản, ổn định hơn khi train.

---

## 6.4. Boundary-texture coupling

Vì texture và biên thường liên hệ mạnh trong food ingredients, thêm nhánh biên:
[
F_{\text{edge}}=\phi_{\text{edge}}(I)
]
[
F_{\text{out}}=\mathcal G(F_{\text{sem}},F_{\text{tex}},F_{\text{edge}})
]

Rất phù hợp với nguyên liệu có ranh giới yếu nhưng texture khác nhau.

---

# 7) Loss function cho texture-aware learning

Nếu chỉ thêm nhánh mà không ép học texture, mô hình có thể bỏ qua nó.
Cần một ràng buộc toán học.

## 7.1. Segmentation loss chuẩn

[
\mathcal L_{\text{seg}} = \mathrm{CE}(\hat Y,Y)
]

---

## 7.2. Boundary loss

Nếu texture giúp biên tốt hơn:
[
\mathcal L = \mathcal L_{\text{seg}} + \lambda \mathcal L_{\text{edge}}
]

---

## 7.3. Texture consistency loss

Buộc vùng cùng lớp có texture feature gần nhau:
[
\mathcal L_{\text{tex}}=
\sum_{(p,q)\in\mathcal P^+}|F_{\text{tex}}(p)-F_{\text{tex}}(q)|^2
------------------------------------------------------------------

\sum_{(p,q)\in\mathcal P^-}\max(0,m-|F_{\text{tex}}(p)-F_{\text{tex}}(q)|)^2
]

Trong đó:

* (\mathcal P^+): cặp pixel cùng lớp
* (\mathcal P^-): cặp pixel khác lớp

Đây là kiểu contrastive learning cho texture.

---

## 7.4. Frequency preservation loss

Muốn output giữ thành phần cao tần:
[
\mathcal L_{\text{freq}} = |\mathcal H(\hat Y)-\mathcal H(Y)|_1
]
với (\mathcal H) là high-pass operator hoặc Laplacian.

---

# 8) Áp vào FoodSeg103 thì nên trích texture gì?

FoodSeg103 khó vì:

* nguyên liệu overlap
* cùng lớp có nhiều appearance
* khác lớp có thể nhìn rất giống nhau
* tín hiệu phân biệt nhiều khi là texture/local appearance hơn là shape toàn cục. 

Nên với ngôn ngữ toán – tin học, bạn có thể xem texture discrimination là bài toán tăng **class separability trong không gian đặc trưng cục bộ**:

Cho (z_p = F_{\text{tex}}(p)), mục tiêu là:
[
|z_p-z_q|\ \text{nhỏ nếu } y_p=y_q
]
và
[
|z_p-z_q|\ \text{lớn nếu } y_p\ne y_q
]

Đặc biệt với các cặp dễ nhầm như:

* potato vs pineapple khi nấu chung
* mushroom vs meat lát
* sauce vs soup residue
  thì feature texture phải mang được:
* độ hạt,
* tính xơ,
* độ bóng,
* độ sần,
* hướng sợi / lớp.

---

# 9) Tóm lại bằng một mô hình hình thức

Một mô hình texture-aware segmentation có thể viết gọn:

[
F_{\text{sp}}=\phi_{\text{sp}}(I)
]
[
F_{\text{ctx}}=\phi_{\text{ctx}}(I)
]
[
F_{\text{tex}}=\phi_{\text{tex}}(I)
]
[
F=\mathcal G(F_{\text{sp}},F_{\text{ctx}},F_{\text{tex}})
]
[
\hat Y = \psi(F)
]
[
\mathcal L = \mathcal L_{\text{seg}} + \lambda_1\mathcal L_{\text{aux}} + \lambda_2\mathcal L_{\text{tex}} + \lambda_3\mathcal L_{\text{edge}}
]

Trong đó:

* (\phi_{\text{sp}}): giữ chi tiết không gian
* (\phi_{\text{ctx}}): lấy ngữ cảnh
* (\phi_{\text{tex}}): trích vi cấu trúc
* (\mathcal G): hợp nhất
* (\psi): decoder / head

Đây là cách viết rất phù hợp để đưa vào phần **phương pháp** hoặc **ý tưởng mô hình** của khóa luận.

---

# 10) Kết luận ngắn

Theo ngôn ngữ toán – tin học:

* **Texture extraction** là bài toán xây đặc trưng từ **thống kê cục bộ** của ảnh.
* Nó có thể dựa trên:

  * đạo hàm/gradient,
  * LBP,
  * GLCM,
  * Gabor,
  * Fourier/Wavelet,
  * hoặc conv features trong CNN.
* Trong deep learning, texture chủ yếu nằm ở:

  * **low-level features**,
  * **high-frequency signals**,
  * **channel correlation statistics**.
* Với **BiSeNet + FoodSeg103**, muốn texture-aware thật sự thì cần bổ sung một ánh xạ (\phi_{\text{tex}}) riêng, thay vì chỉ dựa vào Spatial Path. BiSeNet gốc chủ yếu là **detail-aware + context-aware**, chưa phải texture-aware tường minh.  

Mình có thể viết tiếp cho bạn một mục hoàn chỉnh kiểu báo cáo khoa học: **“Cơ sở toán học của đặc trưng texture và hướng tích hợp vào BiSeNet”**.
