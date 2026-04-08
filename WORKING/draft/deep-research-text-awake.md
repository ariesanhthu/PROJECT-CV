# Nghiên cứu chuyên sâu về tăng cường đặc trưng texture cho mô hình phân đoạn ảnh nhẹ trên FOODSEG103 và so sánh benchmark

## Tóm tắt điều hành

FOODSEG103 là bộ dữ liệu phân đoạn **mức nguyên liệu (ingredient-level)** cho ảnh món ăn, có **7.118 ảnh**, **730 món (dish)**, **103 lớp nguyên liệu** (thực tế nhãn thường tính **kèm nền**), và tổng cộng **42.097 mặt nạ (masks)**; mỗi ảnh trung bình có khoảng **6 nhãn nguyên liệu** và được gán nhãn **pixel-wise**. citeturn24view0turn4view3 Bộ dữ liệu được xây dựng từ ảnh chất lượng cao chọn lọc từ Recipe1M và quá trình gán nhãn được mô tả là **tốn ~1 năm**, có các quy tắc như bỏ qua vùng quá nhỏ, gộp nhãn dễ nhầm lẫn theo thị giác và loại bỏ lớp quá ít mẫu để giảm nhiễu/độ mơ hồ của ranh giới. citeturn4view0turn24view0

Về benchmark, các kết quả “chuẩn” trong hệ sinh thái FOODSEG103 thường báo cáo **mIoU/mAcc/aAcc**; các baseline kiểu CNN (FPN, CCNet) cho mIoU khoảng **27.8–35.5**, trong khi các backbone Transformer lớn (ViT-MLA, FoodSAM, FDSNet) có thể đạt **~45–47+ mIoU**, đổi lại chi phí tính toán rất lớn. citeturn16view2turn11view2turn24view0 Một số công trình gần đây báo cáo mức cao hơn (ví dụ FCN/ResNet-50 đạt mIoU **0.5281** trong một thiết lập huấn luyện riêng; Swin‑TUNA báo cáo **50.56% mIoU** với “trainable params” nhỏ), nhưng cần đặc biệt cẩn trọng vì khác biệt cấu hình/split/định nghĩa “nhẹ” có thể làm **không còn so sánh trực tiếp**. citeturn9view0turn21view0turn20search8

Đối với bài toán “fine‑grained food/ingredient segmentation”, **texture** và **biên (boundary/contour)** là chìa khóa vì nguyên liệu thường có kết cấu tương tự nhau sau chế biến, ranh giới bị che khuất, và nhiều lớp có vùng nhỏ. citeturn4view0turn11view0turn13view1 Nghiên cứu này tổng hợp các nhóm kỹ thuật đã được chứng minh hữu ích cho **tăng cường texture** trong phân đoạn (đa tỉ lệ, nhánh chi tiết, miền tần số: Laplacian/wavelet/FFT, Gabor, attention, giám sát biên, SSL/contrastive, distillation) và thiết kế một **giao thức thí nghiệm** + **các biến thể mô hình nhẹ** có khả năng cạnh tranh trên FOODSEG103.

**Shortlist khuyến nghị (mô hình nhẹ + texture-aware) cho FOODSEG103**:
- **SegFormer‑B0** (nhẹ, backbone 3.7M params; FLOPs thấp) + **mô-đun tần số nhẹ (FcaNet/DCT attention)** + **loss biên (Boundary loss/Lovász)**: cân bằng tốt giữa năng lực biểu diễn và chi phí. citeturn37view0turn39search5turn41search0turn41search1  
- **BiSeNetV2** (nhánh detail/semantic sẵn có) + **đầu vào high‑frequency (Laplacian residual)** theo tinh thần FDSNet + **shape/boundary supervision kiểu Gated‑SCNN (nhẹ hóa)**: rất hợp với ranh giới nguyên liệu nhỏ. citeturn32view1turn15view0turn41search2  
- **Fast‑SCNN** (1.11M params) + **trộn ngữ cảnh miền Fourier (FFC/FFT mixer cỡ nhỏ)** + **lọc texture (Gabor/wavelet tuỳ chọn)**: cực nhẹ, kỳ vọng mIoU thấp hơn nhưng hấp dẫn cho triển khai real‑time. citeturn27view2turn40search0turn40search1turn39search3  

Các chi tiết như “kết quả kỳ vọng” cho một số biến thể là **ước lượng có điều kiện** (vì chưa có số liệu công bố trực tiếp trên FOODSEG103 cho nhiều backbone nhẹ), và được đánh dấu rõ trong bảng đề xuất.

## FOODSEG103: dữ liệu, nhãn và kiểu chú giải

**Quy mô và thống kê cốt lõi**  
FOODSEG103 (phiên bản trong bài giới thiệu benchmark) có **7.118 ảnh**, **730 dish**, **103 ingredient**, **42.097 masks**, kích thước ảnh trung bình khoảng **771×647 px**. citeturn24view0 Bộ dữ liệu đi kèm bản mở rộng FoodSeg154 (9.490 ảnh, 154 ingredient) trong cùng công trình giới thiệu. citeturn24view0

**Split huấn luyện/kiểm tra**  
Thiết lập phổ biến trong paper/benchmark là chia **train 4.983 ảnh** và **test 2.135 ảnh** (xấp xỉ 70/30). citeturn24view0turn4view3 Lưu ý: trong các nghiên cứu sau này có thể xuất hiện biến thể dùng “validation” hoặc chia theo lớp base/novel (open-vocabulary), nên khi benchmark cần ghi rõ **đúng split** và **giao thức**. citeturn13view1turn9view0

**Loại annotation và mức độ “fine-grained”**  
Điểm khác biệt của FOODSEG103 là **pixel-wise mask ở mức nguyên liệu**, thay vì chỉ tách “món ăn vs nền” như nhiều tập dish-level cũ. citeturn24view0turn11view0 Việc gán nhãn được mô tả là công phu và có kiểm soát chất lượng: (i) bỏ qua các vùng rất nhỏ để giảm chi phí/nhiễu, (ii) gộp các nguyên liệu có bề ngoài khó phân biệt, và (iii) loại bỏ các lớp có số ảnh quá ít (ví dụ <5) để giảm long-tail cực đoan; từ đó giảm số lớp từ ~125 xuống **103**. citeturn4view0turn24view0  
Ngoài ra, hệ nhãn có nhóm “**other ingredients**” để hấp thụ phần không chắc chắn/hiếm. citeturn4view3

**Độ mất cân bằng và thách thức thị giác**  
FOODSEG103 được nhấn mạnh là khó do **đa dạng cách chế biến gây intra-class variance** và **phân phối long-tail** của nguyên liệu. citeturn4view0turn11view4 Trong ngữ cảnh này, texture/biên đóng vai trò lớn vì nhiều nguyên liệu khác nhau có thể có **kết cấu tương tự** sau nấu, và ranh giới thường bị che khuất hoặc “hoà” vào nhau. citeturn11view0turn13view1

## Benchmark hiện có trên FOODSEG103 và các khoảng trống khi so sánh mô hình nhẹ

**Metric chính thống**  
Các công trình quanh FOODSEG103 thường báo cáo **mIoU**, **mAcc** và **aAcc** cho semantic segmentation. citeturn24view0turn11view1

**Benchmark nền tảng (baseline và SOTA theo một số nguồn chính)**  
Trong hệ so sánh tổng hợp (có kèm FLOPs/Params) của FDSNet, các kết quả tiêu biểu trên FOODSEG103 gồm (mIoU/mAcc và chi phí):  
- **FPN**: mIoU **27.80**, FLOPs **277.84G**, Params **33.07M**  
- **CCNet**: mIoU **35.50**, FLOPs **615.28G**, Params **71.36M**  
- **ViT/B‑MLA**: mIoU **45.10**, FLOPs **257.12G**, Params **102.59M**  
- **FoodSAM**: mIoU **46.42** (kết hợp baseline + mask SAM), FLOPs **460.13G**, Params **632.75M**  
- **FDSNet(Swin)**: mIoU **47.34**, FLOPs **182.74G**, Params **101.93M** citeturn16view2turn11view2

Các con số baseline “DeepLabV3+” cũng xuất hiện trong thống kê tổng quan của paper gốc (mIoU@DeepLabV3+ **34.2** cho FoodSeg103 trong bảng so sánh dữ liệu), cho thấy baseline CNN truyền thống trên tập này không quá cao nếu không có thiết kế/tiền huấn luyện phù hợp. citeturn24view0

**Các benchmark/thiết lập khác cần thận trọng khi đối chiếu**  
Một bài benchmark năm 2025 (Procedia Computer Science) báo cáo **FCN/ResNet‑50 mIoU 0.5281**, cao hơn đáng kể so với nhiều số mIoU “chuẩn” khác, đồng thời so sánh DeepLabV3 (0.4478) và PSPNet (0.3366), và cho rằng FCN vượt FoodSAM (mIoU 0.464) trong thiết lập của họ. citeturn9view0turn11view2 Điều này gợi ý:
- FOODSEG103 nhạy với **recipe/config huấn luyện**, augmentation, lịch LR, crop size, và cách xử lý class imbalance. citeturn6view0turn9view0  
- Cần công bố rõ: **split**, **backbone**, **kích thước input**, **lịch train**, và **tiền huấn luyện** để tránh “không cùng bài toán”.

**Xu hướng mới: open-vocabulary và PEFT**  
- **OVFoodSeg (CVPR 2024)** đánh giá theo giao thức **base/novel classes**, báo cáo mIoU cho novel/base/all theo nhiều split; ví dụ OVFoodSeg đạt mIoU_all khoảng **~42–43** tùy split, và mIoU_novel tăng đáng kể so với SAN. citeturn13view1  
- **Swin‑TUNA (arXiv 2025)** báo cáo đạt **50.56% mIoU** trên FoodSeg103, với **8.13M “trainable parameters”** (PEFT). citeturn21view0turn17view0 Tuy nhiên, vì PEFT giảm tham số *cập nhật khi fine-tune* chứ không nhất thiết giảm **tham số/thời gian suy luận**, và còn có issue cộng đồng về thống kê tham số trainable khác với mô tả, nên khi coi đây là “lightweight” cần ghi rõ định nghĩa và đo lại trên môi trường chuẩn. citeturn20search8

**Khoảng trống đối với “lightweight segmentation” đúng nghĩa**  
Các họ mô hình real-time/edge (Fast‑SCNN, BiSeNetV2, MobileNet+LR‑ASPP, ERFNet, DDRNet‑light…) rất mạnh về triển khai, nhưng **chưa thấy** được benchmark rộng rãi, có kiểm soát cùng giao thức, trên FOODSEG103 trong các bảng chính thống nêu trên (ít nhất trong các nguồn đã truy vết). Do đó, cần một protocol chuẩn hoá để so sánh “nhẹ thật” trên FOODSEG103.

## Kỹ thuật tăng cường texture cho phân đoạn ảnh nhẹ: khảo sát và phân tích phù hợp với FOODSEG103

Phần này tập trung vào các kỹ thuật đã được chứng minh (trên food segmentation hoặc các bài segmentation nói chung) là giúp tăng “chi tiết/texture/biên”, đồng thời có khả năng tích hợp vào mô hình nhẹ.

**Nhánh chi tiết + hợp nhất đa tỉ lệ (multi-scale / bilateral)**
- Biểu tượng tiêu biểu: BiSeNetV2 thiết kế **Detail Branch** (giữ chi tiết không gian) và **Semantic Branch** (ngữ nghĩa), hợp nhất bằng cơ chế guided aggregation; paper có báo cáo độ phức tạp theo **GFLOPs** và mIoU/FPS trên Cityscapes, minh họa triết lý “giữ chi tiết” cho bài toán real-time. citeturn32view1turn33view0  
- Với FOODSEG103, nhánh detail đặc biệt phù hợp vì nguyên liệu nhỏ yêu cầu biên sắc nét; tuy nhiên cần tránh “overfit texture” khiến nhầm các nguyên liệu có texture giống nhau.

**Nhấn mạnh biên/đường bao (edge/contour supervision)**
- **Gated‑SCNN (ICCV 2019)** đề xuất kiến trúc hai luồng, tách **shape stream** xử lý thông tin biên song song với luồng segmentation chính, và dùng gating để giảm nhiễu; hướng này trực tiếp nhắm tới “sharper boundaries”. citeturn41search2turn41search6  
- **Boundary Loss** (Kervadec et al.) đề xuất loss dựa trên biên/contour (thay vì chỉ vùng), bổ sung cho loss vùng (CE/Dice), đặc biệt hữu ích khi mất cân bằng lớp và khi cần tối ưu biên. citeturn41search0turn41search4  
- **Lovász‑Softmax** tối ưu xấp xỉ trực tiếp mIoU/Jaccard (IoU), thường cải thiện mIoU so với chỉ CE, nhất là khi mIoU là metric chính. citeturn41search1turn41search5  

**Miền tần số: Laplacian / wavelet / DCT / FFT**
- **Laplacian Pyramid & high‑frequency residuals**: FDSNet (2025) dùng **Laplacian Pyramid**, trong đó nhánh nông xử lý **high‑frequency residuals** ở full resolution để tăng chi tiết không gian, nhánh sâu (Swin/ViT) xử lý ảnh downsample để lấy ngữ nghĩa, kèm module hợp nhất đa tỉ lệ. citeturn15view0turn16view2 Đây là một trong những ví dụ “texture/high‑frequency aware” trực tiếp cho food segmentation trên FoodSeg103.  
- **DCT/Frequency Channel Attention**: FcaNet (ICCV 2021) đưa DCT vào channel attention, và chứng minh GAP tương đương một trường hợp đặc biệt của DCT; khung “multi-spectral channel attention” này thường thêm chi phí nhỏ nhưng giúp mô hình nhạy hơn với thành phần tần số. citeturn39search5turn39search17 Trong mô hình nhẹ, FcaNet thường phù hợp vì là mô-đun attention “gọn”, có thể gắn vào backbone MobileNet/SegFormer‑B0/BiSeNet.  
- **Wavelet-based segmentation**: WTUNet (SPIE) mô tả việc tích hợp wavelet transform để khai thác đặc trưng miền tần số nhằm tăng biểu diễn texture. citeturn39search14 Gần đây hơn, FE‑UNet (arXiv 2025) đưa “Deep Wavelet Convolution (DWTConv)” để tăng cường học đặc trưng tần số trong segmentation. citeturn39search10 Với FOODSEG103, wavelet có tiềm năng tốt cho chi tiết “hạt/mảnh” (sesame, pepper…) và biên phức tạp, nhưng cần kiểm soát để không tăng latency quá nhiều.  
- **FFT/Fourier modules**:  
  - **Fast Fourier Convolution (FFC, NeurIPS 2020)** đề xuất toán tử convolution dựa trên Fourier với đặc tính receptive field “non-local” và fusion đa tỉ lệ ngay trong unit. citeturn40search0turn40search3  
  - **SegCFT** đề xuất một kiến trúc segmentation “transformer-like” dùng **FFT-based Context-aware Feature Mixer** và Hierarchical Fourier Transform để giảm chi phí so với self-attention và tăng hiệu quả segmentation. citeturn40search1  
  Những hướng này phù hợp khi mô hình nhẹ thiếu global context; tuy nhiên FFT có thể làm tăng chi phí triển khai trên thiết bị hạn chế nếu không tối ưu kernel.  
- **Distillation theo miền tần số**: FreeKD (CVPR 2024) gợi ý distillation sử dụng “semantic frequency prompt”, là một hướng hợp nhất giữa distillation và feature tần số. citeturn40search22

**Bộ lọc texture truyền thống: Gabor / low-pass / high-pass**
- Một hướng là dùng **Gabor/adaptive Gabor** để bắt texture/hướng (orientation) và tăng kênh đầu vào hoặc nhánh texture; một bài về semantic segmentation với adaptive Gabor filters nhấn mạnh Gabor giúp bắt đặc trưng texture/orientation và cân bằng số tham số–độ chính xác. citeturn39search3 Ngoài ra, AGCNs (Pattern Recognition) cũng là một ví dụ về “adaptive Gabor conv nets”. citeturn39search15  
- Trong FOODSEG103, Gabor có thể hữu ích cho các nguyên liệu có vân/nhịp (noodle, steak fiber, cucumber slices), nhưng rủi ro là tăng nhầm lẫn nếu màu/ánh sáng chi phối mạnh hơn texture.

**Attention không gian–kênh và “context encoding”**
- **EncNet / Context Encoding Module (CVPR 2018)** tập trung vào nắm bắt ngữ cảnh toàn cục và “selectively highlights class-dependent featuremaps” với chi phí tăng thêm nhỏ (“marginal extra computation cost”). citeturn39search4turn39search0 Với FOODSEG103, EncNet hữu ích khi nguyên liệu nhìn giống nhau, cần ngữ cảnh món ăn để quyết định.  
- **CANet (food segmentation, 2023/2024)** mô tả Cross Spatial Attention (CSA) theo trục ngang/dọc để thu ngữ cảnh dài hạn với chi phí thấp hơn attention bậc hai, đồng thời nhấn mạnh cải thiện biên/đa tỉ lệ cho đối tượng thực phẩm. citeturn23view0turn16view2  
- **GourmetNet** dùng multi-scale waterfall features với spatial+channel attention, và mô tả lợi ích về shape/color/texture trong segmentation món ăn. citeturn22search3turn22search11

**SSL/contrastive pretraining cho food domain**
- Một điểm nổi bật trong hệ FOODSEG103 là tiền huấn luyện đa phương thức **ReLeM** (kết hợp recipe/text với ảnh) để giảm intra-class variance do cách nấu. citeturn24view0turn4view0  
- OVFoodSeg cũng dùng pretraining theo recipe (Recipe‑1M+) để học biểu diễn ngữ nghĩa thực phẩm và giúp generalize cho “novel ingredients”. citeturn13view0turn13view1  
Nhìn theo “texture”, SSL/contrastive không trực tiếp là “lọc texture”, nhưng thường tăng khả năng học feature ổn định trước biến thiên ánh sáng/màu, từ đó gián tiếp giúp phân biệt texture tinh.  

## Các họ mô hình segmentation nhẹ đáng benchmark trên FOODSEG103

Dưới đây là các họ mô hình “nhẹ” theo nhiều định nghĩa (ít params, ít FLOPs, hoặc real‑time FPS), kèm phân tích phù hợp FOODSEG103. Một số số liệu params/FLOPs thay đổi theo input; nếu paper không báo cáo, phần đó được đánh dấu “chưa công bố”.

| Họ mô hình | Điểm mạnh liên quan texture/biên | Chi phí (tham khảo) | Nhận xét phù hợp FOODSEG103 |
|---|---|---|---|
| Fast‑SCNN | Thiết kế chia nhánh giữ detail + global context, nhấn mạnh lợi ích quanh **boundary và vật thể nhỏ** | **~1.11M params** (paper Fast‑SCNN); FLOPs không nêu trực tiếp trong paper | Rất hấp dẫn để baseline “siêu nhẹ”; cần bù ngữ cảnh (FFT/ASPP nhẹ) và xử lý class imbalance tốt. citeturn27view2turn26view0 |
| BiSeNetV2 | Detail Branch/ Semantic Branch rõ ràng; ablation có **GFLOPs** và trade-off speed/accuracy | GFLOPs (Cityscapes) khoảng **~21.15** cho cấu hình chính; params không nêu rõ trong arXiv v1 | Rất hợp ingredient boundaries; nên thêm texture/frequency input ở detail branch và loss biên. citeturn32view1turn33view0 |
| ERFNet | Real‑time, kiến trúc gọn, có số params trong bảng so sánh | **~2.1M params** (trong so sánh của Fast‑SCNN); (paper ERFNet nhấn mạnh FPS) | Là baseline CNN tốt cho edge device; có thể gắn FcaNet/EncNet để tăng phân biệt nguyên liệu. citeturn27view2turn38view0 |
| SegFormer‑B0 | Encoder Transformer phân cấp + decoder MLP đơn giản; thường cho biên “mịn hơn” so với một số CNN | SegFormer‑B0: **3.7M params**, **8.4G FLOPs** (bảng efficiency); MiT‑B0 encoder ~0.6 GFLOPs/3.7M params (bảng encoder) | Ứng viên mạnh cho “nhẹ nhưng mạnh”; cần tăng nhạy texture (DCT/wavelet) và xử lý long-tail. citeturn37view0turn37view2turn36view0 |
| MobileNet/ShuffleNet + head nhẹ (LR‑ASPP/DeepLab-lite) | Depthwise separable conv tốt cho chi phí; dễ phối hợp attention nhỏ | Số liệu phụ thuộc bản triển khai; BiSeNetV2 paper có thử MobileNetV2 làm semantic branch nhưng GFLOPs tăng mạnh trong cấu hình đó | Tiềm năng “mobile-grade”; cần kiểm chứng kỹ trên FOODSEG103 do fine-grained khó hơn Cityscapes. citeturn31view2turn32view1 |
| Fourier/Wavelet enhanced U‑Net (nhẹ hoá) | Skip connections tốt cho chi tiết; wavelet tăng khả năng biểu diễn tần số/texture | FE‑UNet mô tả DWTConv; chi phí tuỳ cấu hình | Hợp khi muốn biên sắc và cấu trúc U‑Net; cần tránh decoder quá nặng. citeturn39search10turn39search14 |

Điểm quan trọng: “nhẹ” cho FOODSEG103 không chỉ là params/FLOPs, mà còn là **khả năng giữ chi tiết** ở nguyên liệu nhỏ. Vì vậy, các kiến trúc “hai nhánh” (BiSeNet/Fast‑SCNN) và “multi‑scale + MLP decoder” (SegFormer‑B0) là ứng viên tốt.

## Giao thức thí nghiệm đề xuất cho FOODSEG103

**Mục tiêu**: đánh giá công bằng hiệu quả của mô hình nhẹ và các mô-đun tăng cường texture/biên, trên một giao thức tái lập (reproducible) và có kiểm định thống kê.

**Chuẩn hoá dữ liệu và split**
- Dùng đúng **train/test 4.983/2.135** theo benchmark FOODSEG103 (nếu có file danh sách cố định từ repo/benchmark); nếu không có danh sách cố định, phải **cố định seed + công bố danh sách split** để tái lập. citeturn24view0turn4view3  
- Xử lý nhãn: thống nhất cách tính **số lớp** (103 ingredient hoặc 104 gồm background) và cách xử lý “other ingredients”. citeturn4view3turn16view0  

**Huấn luyện: cấu hình nền tảng (baseline)**
Một cấu hình nền tham chiếu trực tiếp từ paper benchmark FOODSEG103:
- Resize augmentation với tỉ lệ khoảng **0.5–2.0**, random crop **768×768**, random horizontal flip, và color jitter; huấn luyện quy mô ~**80k iterations**, batch ~8; optimizer **SGD momentum 0.9**, weight decay **5e‑4**, poly LR schedule (power 0.9). citeturn6view0turn6view1  
Khuyến nghị: dùng cấu hình này làm baseline cho CNN nhẹ để so sánh công bằng, còn Transformer-lite có thể dùng AdamW nhưng phải báo cáo song song (SGD vs AdamW) nếu cần.  

**Augmentation tăng cường texture (đề xuất)**
Ngoài các augmentation cơ bản ở trên citeturn6view0, thêm các biến đổi nhấn mạnh texture nhưng vẫn “thực tế”:
- Random Gaussian blur nhẹ và random sharpening (mô phỏng nhiễu camera/độ nét khác nhau).
- Random JPEG compression / color temperature jitter (mô tả biến thiên thiết bị/chụp).
- CutMix/MixUp ở mức pixel hoặc patch (thận trọng vì có thể phá cấu trúc món ăn; chỉ dùng cho regularization nhẹ).

**Loss và học đa mục tiêu (texture/biên)**
- Loss nền: Cross‑Entropy (CE). citeturn9view0turn6view0  
- Bổ sung tối ưu mIoU: **Lovász‑Softmax** (fine-tune cuối hoặc kết hợp với CE). citeturn41search1turn41search9  
- Bổ sung biên: **Boundary Loss** kết hợp CE/Dice để tăng chất lượng ranh giới và ổn định khi mất cân bằng. citeturn41search0turn41search12  
- Khi long-tail mạnh: cân nhắc **Focal Loss** (biến thể cho segmentation) để tập trung vào hard examples. citeturn41search19turn41search3  

**Đánh giá (metrics)**
- Metric chính: mIoU, mAcc, aAcc theo truyền thống FOODSEG103. citeturn11view1turn24view0  
- Metric chất lượng biên: dùng boundary F-score hoặc metric biên tương tự; Gated‑SCNN nhấn mạnh cải thiện cả mask mIoU và boundary quality. citeturn41search6turn41search2  
- Báo cáo thêm: per-class IoU, macro‑F1 theo lớp hiếm (để thấy lợi ích texture trên long-tail).

**Kiểm định thống kê**
- Chạy **≥5 seeds** cho mỗi cấu hình (đặc biệt với mô hình nhẹ vốn dao động lớn) và báo cáo mean±std (mIoU và boundary metric). citeturn29search7  
- Dùng kiểm định ghép cặp trên per‑image IoU (paired t-test hoặc Wilcoxon signed‑rank) và/hoặc bootstrap CI 95% cho chênh lệch mIoU.

**Các visualization nên có**
- Qualitative: ảnh gốc + GT + dự đoán (baseline vs texture‑enhanced) tập trung vào nguyên liệu nhỏ và vùng ranh giới nhập nhằng (ví dụ topping nhỏ, salad mix). (FDSNet có minh hoạ trực quan so sánh). citeturn16view1  
- Confusion matrix theo lớp (hoặc top‑K confusion) để thấy nhóm nguyên liệu hay nhầm lẫn.  
- Biểu đồ per‑class IoU (sorted) và “tail IoU” (trung bình trên nhóm lớp hiếm).

```mermaid
flowchart TB
  A[FOODSEG103: images + pixel masks] --> B[Preprocess\n(label mapping, ignore/other ingredients)]
  B --> C[Train split / Test split\n(fixed lists + seeds)]
  C --> D[Baseline training\n(SGD/AdamW + poly)]
  D --> E[Texture enhancements\n(Laplacian/Wavelet/DCT/FFT/Gabor)]
  E --> F[Losses\nCE + Lovasz + Boundary (+Focal)]
  F --> G[Evaluation\nmIoU, mAcc, aAcc, boundary score]
  G --> H[Analysis\nper-class IoU, confusion, ablation, stats tests]
```

## Đề xuất biến thể mô hình kết hợp backbone nhẹ + mô-đun texture, bảng so sánh và shortlist khuyến nghị

### Biến thể đề xuất

**Biến thể A: SegFormer‑B0 + DCT attention (FcaNet) + loss biên**
- Backbone: SegFormer‑B0 (nhẹ; 3.7M params, FLOPs thấp theo bảng efficiency). citeturn37view0turn37view2  
- Texture module: chèn **FcaNet/multi-spectral channel attention** vào các stage sớm–trung (ưu tiên nơi còn giữ high-res feature) để tăng nhạy thành phần tần số. citeturn39search5turn39search17  
- Loss: CE + Lovász‑Softmax + Boundary loss. citeturn41search1turn41search0  
- Kỳ vọng: tăng per‑class IoU cho nhóm nguyên liệu nhỏ/texture mạnh (seeds, herbs) và cải thiện boundary metric.

**Biến thể B: BiSeNetV2 + Laplacian residual input (nhánh detail) + shape supervision**
- Backbone: BiSeNetV2 (đã chứng minh trade-off chi phí theo GFLOPs). citeturn32view1turn33view0  
- Texture module: thêm tiền xử lý **Laplacian Pyramid / high‑frequency residual** và đưa residual vào Detail Branch (tương tự tinh thần FDSNet dùng high‑frequency residuals cho shallow branch). citeturn15view0turn16view2  
- Boundary module: giản lược ý tưởng **shape stream** của Gated‑SCNN thành một nhánh biên rất nông (1–2 blocks depthwise) với supervision cạnh. citeturn41search2turn41search6  
- Kỳ vọng: cải thiện đáng kể “biên bị dính” giữa nguyên liệu.

**Biến thể C: Fast‑SCNN + FFT/Fourier mixer (nhẹ hoá global context)**
- Backbone: Fast‑SCNN (1.11M params) để có baseline cực nhẹ. citeturn27view2turn26view0  
- Texture module: thay hoặc bổ sung vào “global feature extractor” bằng (i) **FFC** (Fourier convolution) hoặc (ii) một biến thể nhỏ của FFT-based mixer theo hướng SegCFT (ưu tiên cấu hình nhỏ, tránh self-attention nặng). citeturn40search0turn40search1  
- Loss: CE + Lovász (tuỳ độ ổn định), có thể thêm Focal nếu lớp hiếm bị bỏ qua. citeturn41search1turn41search19  
- Kỳ vọng: tăng khả năng phân biệt nguyên liệu dựa trên ngữ cảnh toàn cục, bù cho năng lực biểu diễn hạn chế của mô hình siêu nhẹ.

**Biến thể D: MobileNet‑based encoder + EncNet (context encoding) + DCT attention**
- Dùng encoder MobileNet/ShuffleNet (tùy framework) + head nhẹ; thêm **Context Encoding Module (EncNet)** để highlight featuremap theo lớp với chi phí tăng thêm nhỏ. citeturn39search4turn39search0  
- Kết hợp thêm FcaNet tại một số block để tăng texture sensitivity. citeturn39search5  
- Đây là biến thể “mobile‑first” phù hợp triển khai, nhưng cần benchmark vì FOODSEG103 khó hơn segmentation dish-level. citeturn11view0turn24view0

### Bảng so sánh kết quả reported và kết quả kỳ vọng

Bảng dưới đây trộn **(i) số liệu đã báo cáo trực tiếp trên FOODSEG103** và **(ii) kỳ vọng** cho các biến thể “nhẹ + texture” (ước lượng, cần kiểm chứng thực nghiệm). Những ô “—” là **chưa thấy công bố trực tiếp** trong các nguồn đã truy vết.

| Mô hình / biến thể | mIoU FOODSEG103 (reported) | Params / FLOPs (reported) | Nhận xét về “nhẹ” | mIoU kỳ vọng sau texture‑enhance |
|---|---:|---|---|---:|
| FPN | 27.80 | 33.07M / 277.84G citeturn16view2 | Không nhẹ | — |
| ViT/B‑MLA (baseline FoodSAM) | 45.10 | 102.59M / 257.12G citeturn16view2turn11view2 | Nặng | — |
| FoodSAM | 46.42 | 632.75M / 460.13G citeturn16view2turn11view2 | Rất nặng | — |
| FDSNet(Swin) | 47.34 | 101.93M / 182.74G citeturn16view2turn15view0 | Nặng (nhưng “rẻ” hơn FoodSAM) | — |
| Swin‑TUNA (PEFT) | 50.56 | 8.13M trainable params (PEFT) citeturn21view0turn20search8 | “Nhẹ khi fine‑tune”, không chắc “nhẹ khi inference” | — |
| FCN/ResNet‑50 (benchmark 2025) | 0.5281 | — citeturn9view0 | Không rõ (không tập trung lightweight) | — |
| SegFormer‑B0 (gốc) | — | 3.7M / 8.4G (bảng efficiency) citeturn37view0turn37view2 | Nhẹ | ~40–46 (ước lượng phụ thuộc train) |
| SegFormer‑B0 + FcaNet + (Lovász+Boundary) | — | + overhead nhỏ (attention + loss) citeturn39search5turn41search0turn41search1 | Nhẹ | ~43–48 (ước lượng) |
| BiSeNetV2 (gốc) | — | GFLOPs ~21.15 (Cityscapes); params không nêu citeturn32view1turn33view0 | Nhẹ/real‑time | ~38–45 (ước lượng) |
| BiSeNetV2 + Laplacian residual + shape supervision | — | + overhead nhỏ–vừa citeturn15view0turn41search2 | Nhẹ/real‑time | ~42–47 (ước lượng) |
| Fast‑SCNN (gốc) | — | 1.11M params citeturn27view2 | Rất nhẹ | ~30–40 (ước lượng) |
| Fast‑SCNN + FFC/FFT mixer | — | tăng FLOPs tùy mức FFT citeturn40search0turn40search1 | Vẫn nhẹ nếu tiết chế | ~33–42 (ước lượng) |

**Lưu ý quan trọng về tính so sánh**: Các “mIoU kỳ vọng” ở nhóm mô hình nhẹ là dự báo dựa trên xu hướng từ các kỹ thuật tăng texture/biên đã chứng minh hiệu quả trong segmentation nói chung (FcaNet, boundary loss, Fourier/wavelet, shape stream…) và đặc thù FOODSEG103; cần thí nghiệm ablation để xác nhận.

### Shortlist khuyến nghị triển khai thí nghiệm

1) **SegFormer‑B0 + FcaNet + Lovász/Boundary**: ưu tiên nếu mục tiêu là “nhẹ nhưng mIoU cao”, cân bằng tốt giữa ngữ cảnh và chi tiết. citeturn37view0turn39search5turn41search0turn41search1  
2) **BiSeNetV2 + Laplacian residual (detail) + boundary supervision**: ưu tiên nếu mục tiêu là biên sắc nét, real-time, và dễ triển khai trên thiết bị. citeturn15view0turn32view1turn41search2  
3) **Fast‑SCNN + Fourier mixer mini (FFC/SegCFT‑style)**: ưu tiên nếu mục tiêu “siêu nhẹ”, chấp nhận mIoU thấp hơn nhưng kiểm chứng được lợi ích của miền tần số trong điều kiện compute hạn chế. citeturn27view2turn40search0turn40search1  

```mermaid
graph LR
  subgraph Backbones_nhe
    A[Fast-SCNN]
    B[BiSeNetV2]
    C[SegFormer-B0]
  end

  subgraph Texture_Modules
    T1[Laplacian / high-frequency residual]
    T2[Wavelet (DWTConv)]
    T3[DCT attention (FcaNet)]
    T4[FFT/Fourier (FFC / mixer)]
    T5[Gabor branch]
  end

  subgraph Supervision
    S1[Boundary Loss]
    S2[Lovasz-Softmax]
    S3[Shape stream (lite)]
  end

  A --> T4
  A --> S1
  B --> T1
  B --> S3
  C --> T3
  C --> S2
```

---

### Ghi chú những chi tiết chưa xác định

Một số hạng mục người đọc cần tự xác nhận khi triển khai vì các nguồn hiện có không thống nhất/không công bố:
- Nhiều mô hình “nhẹ” (MobileNet‑DeepLab‑lite, DDRNet‑light, UNet‑Mobile) **chưa có số mIoU chính thức trên FOODSEG103** trong các bảng chuẩn đã truy vết; do đó bắt buộc tự benchmark theo protocol thống nhất.  
- “Nhẹ” theo PEFT (Swin‑TUNA) cần phân biệt **trainable params** và **tổng chi phí suy luận**; có dấu hiệu không nhất quán trong thống kê trainable params ở issue repo, nên cần đo lại. citeturn20search8turn21view0