5) Dataset nên xử lý thế nào: thứ tự ưu tiên thực tế
A. Dọn taxonomy trước

Việc class seaweed = 0 pixel trong rebalanced là tín hiệu xấu. Class như vậy chỉ làm logits thừa, loss nhiễu, metric méo. Nên:

bỏ khỏi ontology train, hoặc
xác nhận lại pipeline remap/mask generation xem class này đáng ra phải merge/drop hẳn.

Cái này nên làm ngay.

B. Audit theo split, không chỉ toàn dataset

Bạn cần thống kê lại cho train/val riêng:

presence_count từng class
total_pixels từng class
số class chỉ có 1–2 ảnh trong train
số class vắng hoàn toàn ở val hoặc train

Vì bài toán của bạn hỏng chủ yếu ở trainability. File train guidance trước đó cũng đã nói rất rõ: trước khi đổi backbone hay thêm module, phải kiểm tra split coverage và class chết trong train.

C. Chia nhóm lớp để xử lý khác nhau

Dựa trên texture + size + frequency, mình chia thành 4 nhóm:

Nhóm 1: stable, đủ dữ liệu, object lớn
bread, chicken duck, potato, pork, steak, rice, broccoli, sauce...
Nhóm này train baseline được.

Nhóm 2: hard pair texture-driven
fish / chicken duck / pork / pie / onion / spring onion / blueberry / cherry...
Nhóm này cần pair-aware sampling.

Nhóm 3: object nhỏ, dễ mất khi crop
dried cranberries, blueberry, nut, onion, vegetable, tomato, tea...
Nhóm này cần foreground-aware crop và hạn chế downscale quá mạnh.

Nhóm 4: quá hiếm / không đủ thống kê tin cậy
những lớp patch quá ít như pudding 5 patch, date 5, fig 5, salad 10...
Với nhóm này, đừng vội kết luận texture descriptor là “bản chất lớp”; sample ít quá. Nên hoặc tiếp tục gộp, hoặc giảm trọng số đánh giá khi phân tích.

D. Crop phải đổi theo texture

Actionables của summary cho thấy:

16 class cần larger crop
53 class cần edge_detail_branch
40 class cần group_sampling

Mình đồng ý với hướng này, nhưng ưu tiên nên là:

1. larger crop / foreground-aware crop trước
vì hiện rất nhiều class nhỏ hoặc patch ambiguity do crop quá sát.

2. group sampling theo hard pairs
để model gặp nhầm lẫn thực sự thường xuyên hơn.

3. edge/detail branch sau
vì branch thêm vào khi pipeline dữ liệu chưa ổn thường khó biết nó giúp thật hay chỉ overfit.

E. Sampling nên theo pair, không chỉ theo class

Thay vì oversample class hiếm chung chung, nên có 2 sampler:

rare-class sampler
hard-pair sampler

Ví dụ batch có chủ đích trộn:

pork + steak
rice + potato
mushroom-related
leafy greens
fish + chicken duck
vì mục tiêu của bạn không chỉ là “thấy class hiếm”, mà là “học ranh giới giữa hai class dễ nhầm”.
F. Annotation audit: chưa thấy bằng chứng mạnh là lỗi nhãn hàng loạt

Summary đang để annotation_audit = 0. Mình đồng ý là hiện tại chưa có bằng chứng cho việc nhãn sai hàng loạt.

Nhưng với các gallery như rice–potato, mushroom-related, leafy greens, có một số patch rất mờ/zoom-in mạnh. Cái cần audit không hẳn là nhãn sai, mà là:

patch đó có đủ ngữ cảnh để dùng trong texture analysis không
crop generation có đang tạo quá nhiều patch “không nhận diện được cả với người” không

Tức là nên audit patch extraction policy trước khi audit annotation toàn bộ.

6) Từ kết quả này, có nên clean lại dataset không?

Không nên clean toàn bộ lại từ đầu ngay.
Mình sẽ chia như sau:

Nên clean / chỉnh ngay
bỏ class chết
rà lại mapping sau rebalance
kiểm tra split coverage train/val
lọc hoặc gắn cờ các patch texture quá mờ, quá phẳng, quá ít foreground
rà một vòng các hard pair gallery để xem có patch nào extraction vô nghĩa
Chưa nên làm ngay
relabel hàng loạt
đổi taxonomy lớn lần nữa
drop thêm nhiều class chỉ vì khó

Vì hiện tại vấn đề chính chưa phải “dataset bẩn”, mà là:
dataset fine-grained + imbalanced + ambiguous by design.

7) Liên hệ trực tiếp với BiSeNet baseline

BiSeNet paper nhấn mạnh đúng 2 thứ:

phải giữ spatial detail
đồng thời có context path đủ receptive field

Kết quả texture của bạn càng cho thấy FoodSeg cần đúng triết lý này, nhưng với một chỉnh sửa quan trọng:

SP/CP là cần, nhưng vẫn chưa đủ nếu crop/sampling làm mất texture discriminative.

Nên cho project này, thứ tự hợp lý là:

giữ baseline BiSeNet nhẹ
sửa dataset handling
thêm texture-aware ở mức data + supervision + sampling trước
chỉ sau đó mới thêm module kiến trúc
8) Mình chốt plan hành động
Pha 1: xử lý dataset
bỏ class dead
xuất bảng train_presence_count, val_presence_count, train_pixels, val_pixels
tạo danh sách:
rare classes
hard pairs
small-object classes
thêm cờ chất lượng patch: blur, fg_ratio thấp, entropy thấp
Pha 2: sửa pipeline train
foreground-aware crop làm mặc định
thêm larger-crop policy cho small-object / ambiguous classes
sampler theo hard pair + rare class
báo cáo metric thêm:
mIoU_fg_present
per-class IoU cho hard pairs
pair-confusion matrix
Pha 3: chỉ khi 2 pha trên ổn
edge/detail branch cho nhóm leafy greens, onion/spring onion, mushrooms
contrastive/pair loss nhẹ cho hard pairs
texture auxiliary head nhỏ, không quá nặng
9) Kết luận cuối

Dataset của bạn sau phân tích texture cho thấy:

đúng là có vấn đề texture overlap thật
rebalance là hợp lý nhưng chưa đủ
chưa có dấu hiệu cần clean toàn bộ lại
cần xử lý theo hard-pair-aware data strategy trước, rồi mới tới module texture-aware

Nói gọn trong một câu:

Với FoodSeg của bạn, hướng đúng không phải “clean mạnh hơn”, mà là tổ chức lại dữ liệu huấn luyện theo rare class + hard pair + small object + detail preservation.

Nếu bạn muốn, mình sẽ chuyển toàn bộ phân tích này thành một bản guide v2 rất cụ thể, gồm:

bảng phân nhóm class
rule crop theo class
rule sampling theo hard pair
checklist clean/audit
thứ tự experiment 1 → 2 → 3 để train bản v2.