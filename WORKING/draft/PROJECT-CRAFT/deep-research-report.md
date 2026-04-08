# Thực nghiệm DeepLabV3 cho phân đoạn ngữ nghĩa trên FoodSeg103 với backbone lightweight hướng tới triển khai di động

## Mục tiêu và nhiệm vụ nghiên cứu

### Mục tiêu

Triển khai mô hình **DeepLabV3** cho bài toán phân đoạn ngữ nghĩa trên tập dữ liệu **FoodSeg103** (phân đoạn theo thành phần/ingredient ở mức pixel). FoodSeg103 được xây dựng để hỗ trợ các ứng dụng liên quan sức khỏe như ước lượng calo/dinh dưỡng, nhưng bài toán khó do đặc trưng thị giác phức tạp của thực phẩm (nguyên liệu chồng lấp, cùng một nguyên liệu có thể có hình thái rất khác nhau; hoặc hai nguyên liệu khác nhau có thể trông khá giống). citeturn0search2turn2view2turn2view0

Mở rộng:

- Thay đổi backbone theo hướng **lightweight** (ưu tiên họ MobileNet) để đánh giá sự thay đổi về **độ chính xác** và **tốc độ suy luận**, hướng tới khả năng chạy được trên thiết bị di động. MobileNetV2/V3 được thiết kế cho on-device computer vision, sử dụng các khối hiệu quả như depthwise separable convolution (MobileNetV2) và tối ưu latency trên CPU điện thoại (MobileNetV3). citeturn1search0turn1search5  
- Phân tích sự tương quan giữa **độ phân giải dữ liệu đầu vào** và hiệu năng của mô hình **nhằm xác định điểm cân bằng tối ưu trong bài toán phân đoạn ngữ nghĩa** khi ràng buộc bởi latency/bộ nhớ (đặc biệt quan trọng với triển khai di động). citeturn2view1

Định hướng câu hỏi nghiên cứu (đặt trong bối cảnh proposal cần “what/why/how” rõ ràng):

- Với FoodSeg103, backbone lightweight nào (ví dụ MobileNetV2 vs MobileNetV3) giúp đạt **trade-off mIoU–latency** tốt nhất cho DeepLabV3? citeturn1search0turn1search5turn5search1  
- Khi thay đổi độ phân giải đầu vào, hiệu năng (mIoU) giảm theo mức nào, và latency cải thiện ra sao để lựa chọn cấu hình triển khai hợp lý? citeturn2view1  

### Nhiệm vụ cụ thể

1. **Khảo sát tài liệu**  
   Nghiên cứu các survey/nguồn nền về semantic segmentation và các hướng mô hình hoá ngữ cảnh đa tỉ lệ. Trọng tâm là DeepLabV3 (atrous convolution + mô-đun gom ngữ cảnh đa tỉ lệ như ASPP) và các biến thể/thiết kế liên quan đến tối ưu hoá tốc độ. citeturn5search1turn0search35  
   Song song, khảo sát FoodSeg103: mục tiêu xây dựng benchmark ingredient-level segmentation, nguồn ảnh (Recipe1M), đặc điểm phân bố lớp và các khó khăn đặc thù của ảnh món ăn. citeturn2view0turn6search15turn0search2

2. **Chuẩn bị dữ liệu**  
   Xây dựng pipeline xử lý dữ liệu từ tập FoodSeg103, bao gồm tiền xử lý ảnh/mask, chuẩn hoá nhãn (lưu ý: thường mô tả “103 ingredient categories”, nhưng thực tế nhãn có **background** nên tổng số id có thể là 104 lớp tính cả nền). citeturn6search15turn3view0  
   Thay đổi dữ liệu với các mức phân giải khác nhau để phục vụ thực nghiệm trade-off accuracy–speed (ví dụ 256/320/384/512, tuỳ giới hạn thiết bị). Cơ sở của bước này là yêu cầu proposal phải thể hiện “phương pháp & tính khả thi theo timeline”. citeturn2view1

3. **Xây dựng mô hình baseline**  
   Triển khai và huấn luyện mô hình baseline DeepLabV3 theo thiết lập tái lập được (seed, schedule, augmentations, cấu hình backbone chuẩn như ResNet để làm mốc chất lượng). DeepLabV3 là hệ mô hình khai thác atrous convolution để điều chỉnh field-of-view và mô-đun đa tỉ lệ nhằm cải thiện phân đoạn mà không cần hậu xử lý kiểu DenseCRF trong các cấu hình điển hình. citeturn5search1turn0search7  
   Đồng thời huấn luyện các biến thể DeepLabV3 với backbone lightweight (ưu tiên MobileNet). Việc “xây baseline trước” và “ablation theo bước” bám đúng yêu cầu trình bày objectives theo thứ tự logic trong proposal. citeturn2view1

4. **Đề xuất phương pháp cải tiến**  
   - **boundary refinement**: tập trung cải thiện vùng biên (ranh giới nguyên liệu thường mờ/khó phân định khi nguyên liệu trộn lẫn). Có thể tiếp cận theo 2 hướng: (i) bổ sung decoder kiểu DeepLabV3+ để phục hồi biên tốt hơn; hoặc (ii) thêm loss/nhánh biên (boundary-aware) để ép mô hình khớp ranh giới. citeturn5search0turn6search1turn6search9  
   - *test-time augmentation*: áp dụng các biến đổi hình học đơn giản (flip/scale/rotation tuỳ cấu hình) ở thời điểm suy luận và gộp dự đoán nhằm tăng ổn định dự đoán. Cần đánh giá kèm chi phí latency vì TTA làm tăng số lượt inference. citeturn6search12turn6search8

5. **Thực nghiệm và đánh giá**  
   So sánh hiệu năng các mô hình thông qua **mean Intersection-over-Union (mIoU)** (thước đo dựa trên tỉ lệ giao/hiệp của mask theo lớp, rồi lấy trung bình), kèm pixel accuracy và các chỉ số triển khai (latency, FPS, kích thước mô hình). citeturn6search18turn6search2

6. **Phân tích kết quả**  
   Đánh giá tác động của từng thay đổi: backbone, độ phân giải đầu vào, boundary refinement, TTA. Phân tích riêng 2 nhóm mục tiêu: (i) chất lượng phân đoạn trên FoodSeg103; (ii) khả năng triển khai di động (latency/bộ nhớ). citeturn2view1  

## Đóng góp dự kiến

Nghiên cứu này dự kiến mang lại các đóng góp sau (được viết theo hướng “expected research contribution” trong research proposal: nêu rõ khoảng trống và giá trị mang lại). citeturn2view1

Thứ nhất, **thực nghiệm có hệ thống kiến trúc DeepLabV3 trên FoodSeg103** (tập benchmark ingredient-level) để tạo mốc tái lập và rút ra cấu hình “mạnh–ổn định” theo điều kiện augment/schedule nhất quán. FoodSeg103 có 7,118 ảnh và được chú giải ở mức pixel theo ingredient; dữ liệu được tuyển chọn từ Recipe1M và tinh chỉnh bởi người gán nhãn. citeturn6search15turn2view0turn0search2

Thứ hai, **thử nghiệm kiến trúc DeepLabV3 với backbone MobileNet** (MobileNetV2/MobileNetV3) nhằm cải thiện tốc độ inference, hướng tới triển khai trên thiết bị di động. MobileNetV2/V3 là các họ mạng được thiết kế cho on-device vision, đồng thời các tác giả MobileNetV2 cũng thảo luận cách xây mô hình segmentation “Mobile DeepLabv3” (giảm/điều chỉnh head) để đạt hiệu quả tính toán tốt hơn. citeturn1search0turn1search5

Thứ ba, **phân tích điểm yếu của dataset** FoodSeg103 dưới góc nhìn mô hình hoá và triển khai: dataset có hiện tượng mất cân bằng khi một số lớp phổ biến chiếm tỉ lệ lớn, trong khi nhiều lớp hiếm ít mẫu; ngoài ra ảnh món ăn có biến thiên mạnh về ánh sáng/góc chụp/bối cảnh, và nguyên liệu có thể nhỏ hoặc bị che khuất. citeturn2view0

Thứ tư, **xác định điểm cân bằng tối ưu giữa độ phân giải và hiệu năng của mô hình**, đặt trong ràng buộc thực tế “accuracy–latency–model size” để có khuyến nghị cấu hình chạy di động. Đây là phần giúp proposal “make a case for significance” theo nghĩa giải quyết nhu cầu triển khai thật, không chỉ benchmark offline. citeturn2view1turn1search3

## Phương pháp nghiên cứu

### Dữ liệu

Sử dụng tập FoodSeg103. Theo mô tả phổ biến, FoodSeg103 có **7,118 ảnh** với **103 nhóm nguyên liệu (ingredient categories)** được gán nhãn theo pixel; dữ liệu là mẫu được tuyển chọn từ Recipe1M và tinh chỉnh bởi người gán nhãn. citeturn6search15turn2view0turn0search2  
Về chia tập, nhiều nguồn thực nghiệm công bố bộ **train 4,983 ảnh** và một split còn lại **2,135 ảnh** (được gọi là validation hoặc test tuỳ quy ước toolkit). citeturn6search15turn4view0turn4view3  
Về nhãn, danh sách id thường bao gồm **background** (id 0) cùng các lớp thực phẩm/đồ uống; vì vậy tổng số id có thể là 104 nếu tính cả nền. citeturn4view3turn3view0

Pipeline tiền xử lý và tăng cường dữ liệu (đặt mục tiêu kiểm soát được để làm ablation):

- resize và random crop (quy định rõ kích thước crop theo thí nghiệm độ phân giải)  
- horizontal flip  
- color jitter (hữu ích với biến thiên ánh sáng/bối cảnh trong ảnh món ăn) citeturn6search4turn2view0  
- multi-scale training (nếu dùng, cần cố định danh sách scale và báo cáo ảnh hưởng tới latency/quality)

Các bước này cần được “đóng gói” thành pipeline tái lập để đảm bảo so sánh backbone là công bằng.

### Mô hình cơ sở

Mô hình baseline sử dụng:

- **DeepLabV3**

DeepLabV3 khai thác **atrous (dilated) convolution** để kiểm soát field-of-view và giữ độ phân giải đặc trưng, đồng thời dùng mô-đun đa tỉ lệ (như ASPP theo nhiều atrous rates) để tổng hợp ngữ cảnh đa tỉ lệ cho semantic segmentation. citeturn5search1turn0search35turn0search7

Thiết kế backbone:

- Baseline “mốc chất lượng”: backbone chuẩn (ví dụ ResNet) để đạt chất lượng tốt và làm tham chiếu.  
- Biến thể “hướng di động”: backbone **MobileNetV2** và **MobileNetV3**. MobileNetV2 dựa trên inverted residual + linear bottleneck và dùng depthwise convolution để giảm chi phí tính toán; MobileNetV3 được tối ưu theo hướng hardware-aware và có hai biến thể Large/Small cho các mức tài nguyên khác nhau. citeturn1search0turn1search5

Lý do chọn MobileNet cho mục tiêu di động: thay backbone là can thiệp “đúng điểm” vì backbone thường chiếm phần lớn FLOPs/latency; do đó có thể tạo khác biệt rõ rệt về tốc độ suy luận mà vẫn giữ head DeepLabV3 để so sánh nhất quán theo kiến trúc. citeturn1search0turn5search1

### Các hướng cải tiến đề xuất

#### Boundary refinement

Bổ sung các khối xử lý biên nhằm:

- cải thiện độ chính xác của ranh giới đối tượng/nguyên liệu  
- giảm lỗi phân đoạn ở vùng biên (đặc biệt khi nguyên liệu chồng lấp hoặc texture tương tự nhau)

Hai lựa chọn thực nghiệm (ưu tiên đơn giản, dễ ablation):

- Dùng decoder kiểu **DeepLabV3+** để “recover boundaries” khi upsample và trộn đặc trưng nông–sâu. DeepLabV3+ được mô tả là mở rộng DeepLabV3 bằng một decoder đơn giản nhưng hiệu quả để refine kết quả, nhất là dọc biên đối tượng. citeturn5search0turn5search12  
- Thêm boundary-aware loss/post-processing: ví dụ các họ boundary loss (khi dữ liệu có mất cân bằng vùng/biên) hoặc module refine biên kiểu SegFix (model-agnostic) để đánh giá hiệu quả cải thiện biên mà ít làm nặng backbone. citeturn6search29turn6search9turn6search1

#### Test-time Augmentation

Áp dụng kỹ thuật tăng cường dữ liệu tại thời điểm suy luận (như lật/đổi tỷ lệ; có thể cân nhắc rotation tuỳ chi phí) nhằm:

- cải thiện độ ổn định và độ chính xác tổng thể của các dự đoán  
- giảm sai sót do biến đổi góc nhìn/kích thước đầu vào

Tuy nhiên, do TTA làm tăng số lần chạy mô hình, cần thiết kế thí nghiệm “accuracy gain vs latency cost” rõ ràng, nhất là trong mục tiêu chạy di động. citeturn6search12turn6search8

#### Tối ưu hoá inference hướng tới thiết bị di động

Bên cạnh thay backbone, nghiên cứu sẽ thực nghiệm chuỗi triển khai on-device theo chuẩn hiện hành của PyTorch:

- Xuất và chạy mô hình bằng **ExecuTorch** (stack on-device inference của PyTorch), hỗ trợ triển khai trên Android/iOS và các backend tối ưu. citeturn1search3turn1search35turn1search23  
- (Tuỳ giai đoạn) thử nghiệm quantization để giảm kích thước mô hình và cải thiện latency, vì post-training quantization được mô tả có thể giảm kích thước và cải thiện latency CPU/hardware accelerator với mức giảm chính xác nhỏ (cần kiểm chứng trên segmentation). citeturn5search14turn5search3turn5search25  

### Đánh giá

Các mô hình sẽ được đánh giá dựa trên:

- **mean Intersection-over-Union (mIoU)**: IoU = TP/(TP+FP+FN) theo lớp; mIoU là trung bình IoU trên các lớp (cần quy định có/không ignore background và có/không ignore class vắng mặt). citeturn6search18turn6search2  
- pixel accuracy (bổ sung để đọc nhanh nhưng không thay thế mIoU khi có mất cân bằng lớp)  
- tốc độ suy luận: latency trung bình (ms/ảnh), FPS; đo trên (i) môi trường dev; và (ii) thiết bị di động qua ExecuTorch để phản ánh đúng mục tiêu triển khai. citeturn1search3turn1search35  
- chỉ số triển khai: kích thước mô hình (MB), số tham số, bộ nhớ peak (ước lượng theo thiết bị)

## Kế hoạch thực hiện

Kế hoạch dưới đây được viết theo đúng tinh thần “work plan” trong research proposal: nêu mốc công việc để chứng minh tính khả thi theo thời gian và bám theo objectives. citeturn2view1  
(Mốc thời gian theo ngày dương lịch, phù hợp bối cảnh hiện tại là 15/03/2026.)

| Thời gian (2026) | Nội dung |
| --- | --- |
| 16/03–22/03 | Tìm hiểu đề tài, khảo sát tài liệu, xác định bài toán trên FoodSeg103; khoanh vùng pipeline triển khai DeepLabV3 và tiêu chí đo latency trên thiết bị di động. citeturn2view1turn5search1turn2view0 |
| 23/03–30/03 | Chuẩn bị dataset FoodSeg103, kiểm tra nhãn (background + ingredient classes), dựng dataloader/augmentation; chuẩn hoá kịch bản resize/crop theo các mức độ phân giải dự kiến. citeturn6search15turn4view3 |
| 31/03–13/04 | Xây dựng baseline DeepLabV3 (backbone chuẩn) và đánh giá mIoU/pixel accuracy; thiết lập quy trình log/seed để phục vụ ablation. citeturn5search1turn6search18 |
| 14/04–27/04 | Thay đổi backbone theo hướng lightweight (MobileNetV2/MobileNetV3) và so sánh hiệu năng; thực nghiệm độ phân giải đầu vào để tìm điểm cân bằng accuracy–speed. citeturn1search0turn1search5turn2view1 |
| 28/04–04/05 | Thêm boundary refinement và TTA (có kiểm soát); đánh giá mức cải thiện mIoU và chi phí latency tương ứng. citeturn5search0turn6search12 |
| 05/05–12/05 | Tối ưu và benchmark trên thiết bị di động: xuất mô hình và chạy bằng ExecuTorch, đo latency/FPS và bộ nhớ; (nếu phù hợp) thử quantization để so sánh. citeturn1search3turn5search14 |
| 13/05–17/05 | Hoàn thiện báo cáo: tổng hợp bảng kết quả, phân tích ablation, rút ra khuyến nghị cấu hình triển khai. citeturn2view1 |

## Tài nguyên và đối chiếu yêu cầu viết research proposal

### Tài nguyên

**Dữ liệu**  
FoodSeg103: 7,118 ảnh, có mask theo pixel; chia 2 split (4,983 và 2,135 ảnh) tuỳ quy ước train/val hoặc train/test; dung lượng đóng gói ở định dạng phổ biến có thể ở mức ~1.25GB. citeturn6search15turn4view0turn4view3

**Phần cứng**  
Tối thiểu cần 1 máy có GPU để huấn luyện segmentation ở độ phân giải trung bình; và ít nhất 1 thiết bị di động Android hoặc iOS để benchmark latency triển khai (đặc biệt khi mục tiêu là chạy on-device). Yêu cầu này bám theo mục “Resources” trong proposal (cần nêu rõ thiết bị/tài nguyên cần để dự án khả thi). citeturn2view1turn1search3

**Phần mềm**  
PyTorch cho huấn luyện; công cụ triển khai on-device bằng **ExecuTorch** để xuất và chạy mô hình trên thiết bị di động. citeturn1search3turn1search23

**Benchmark / nguồn chuẩn tham chiếu kết quả**  
FoodSeg103 có repo benchmark kèm bảng mIoU/mAcc cho một số baseline (FPN/CCNet/ViT/Swin...) giúp định vị mức khó của dataset; đây là tham chiếu để so sánh tương đối khi triển khai DeepLabV3 (dù kiến trúc khác). citeturn3view0turn3view1

### Đối chiếu với hướng dẫn của University of Sydney

Theo hướng dẫn “How to write a research proposal”, một proposal cần làm rõ: aims/objectives, expected research contribution, proposed methodology, work plan, resources và bibliography. citeturn2view1  
Bản viết theo flow hiện tại đã đáp ứng trực tiếp các khối nội dung này như sau: (i) **Mục tiêu và nhiệm vụ nghiên cứu** tương ứng aims/objectives và được sắp theo trình tự logic; (ii) **Đóng góp dự kiến** nêu expected research contribution và lý do giá trị của nghiên cứu; (iii) **Phương pháp nghiên cứu** mô tả dữ liệu–mô hình–đánh giá và nêu lý do lựa chọn backbone/triển khai di động; (iv) **Kế hoạch thực hiện** thể hiện work plan và tính khả thi; (v) **Tài nguyên** nêu rõ dữ liệu, phần cứng/phần mềm phục vụ triển khai. citeturn2view1  
Phần **bibliography** được thể hiện ngầm qua hệ thống trích dẫn trong báo cáo; nếu cần đúng “format proposal nộp trường”, có thể chuyển các trích dẫn này thành danh mục tài liệu tham khảo theo chuẩn IEEE/ACM/APA. citeturn2view1

### Tài liệu tham khảo cốt lõi

Chen et al., “Rethinking Atrous Convolution for Semantic Image Segmentation” (DeepLabV3). citeturn5search1  
Wu et al., “A Large-Scale Benchmark for Food Image Segmentation” (FoodSeg103, ReLeM). citeturn0search2turn2view2  
Sandler et al., “MobileNetV2: Inverted Residuals and Linear Bottlenecks” (MobileNetV2, Mobile DeepLabV3). citeturn1search0  
Howard et al., “Searching for MobileNetV3” (MobileNetV3, tối ưu on-device). citeturn1search5  
ExecuTorch Documentation (triển khai PyTorch trên thiết bị di động/edge). citeturn1search3turn1search35