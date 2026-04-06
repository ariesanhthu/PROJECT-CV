# 1) Texture descriptives

lớp mịn nhất: pudding
lớp hạt nhất: salad
lớp nhiều biên vụn nhất: cashew
hard pair mạnh nhất: chicken duck vs fish, similarity texture 0.9959 nhưng color similarity chỉ 0.1768, tức là khó do texture thật, không phải chỉ do màu.

```
Một là, có những lớp texture rất ổn định, dễ học theo bề mặt.
Ví dụ pudding, coffee, olives, chocolate, cherry khá “mượt”.

Hai là, có những lớp bản chất texture cực nhiễu hoặc nhiều cạnh vụn.

Ví dụ salad, lamb, shellfish, bean sprouts, pork, fish;

còn nhóm edge vụn cao có cashew, almond, spring onion, onion, cilantro mint. 

Đây là nhóm rất dễ hỏng khi crop nhỏ, resize mạnh, hoặc backbone làm mất detail sớm.

Ba là, có một số cặp không thể giải bằng “semantic global” đơn thuần.
Ví dụ:

chicken duck vs fish
fish vs pork
onion vs spring onion
blueberry vs cherry
cake vs lamb

Các cặp này texture gần nhau tới mức nếu mask/crop không giữ được chi tiết cục bộ thì model rất dễ trượt.
```

# 2) Texture embedding: không thấy cluster tách rõ, nghĩa là overlap lớn

- texture riêng lẻ không đủ để phân lớp toàn bộ, nhưng dù vậy texture vẫn là tín hiệu rất quan trọng cho một số cặp khó.

- vì vậy hướng đúng không phải “dùng texture thay semantic”, mà là:
giữ baseline semantic nhẹ + thêm cơ chế bảo toàn detail/edge/patch cho hard pairs

Nói cách khác, kết quả này ủng hộ proposal của bạn về hướng texture-aware lightweight segmentation, chứ không ủng hộ việc ném toàn bộ bài toán sang một texture branch nặng. Proposal của bạn mô tả đúng bài toán: nhiều lớp thực phẩm khác nhau ở tín hiệu cục bộ hơn là hình dạng tổng thể.

# 3) Hard-pair analysis: 

`có 2 kiểu nhầm, và phải xử lý khác nhau`

## Kiểu A: texture-driven thật -> Đây là nhóm phải ưu tiên nhất.

Ví dụ từ summary:

    - chicken duck vs fish
    - fish vs pork
    - chicken duck vs pork
    - onion vs spring onion
    - blueberry vs cherry

Các cặp này `similarity texture rất cao`, `color không đủ cứu`.

Xử lý:
-> cần sampling, crop, metric và loss nhắm đúng vào hard pair.

Kiểu B: mixed hoặc bị context/màu chen vào

Ví dụ:

    - fish vs pie
    - rice vs potato
    - pork vs steak
    - leafy greens
    - mushroom-related

Nhìn gallery là thấy một số patch quá zoom-in, blur, hoặc gần như mất hình thái; lúc đó class bị quyết định bởi bề mặt chung + màu + hoàn cảnh món ăn. Đây không hẳn annotation sai, mà là patch ambiguity.

Xử lý:
-> không nên ép model học từ patch quá nhỏ; phải tăng ngữ cảnh cục bộ vừa đủ.

# 4) Dataset-level reality

rebalance đã giúp, nhưng chưa giải quyết hết “trainability”
Bản rebalanced có 76 foreground class + background, tổng logits 77. Nó đã merge/drop một số lớp hiếm cực đoan để giảm mất cân bằng

Nhưng số liệu cho thấy sau rebalance vẫn còn:

class chết: seaweed = 0 pixel, 0 presence
nhiều lớp xuất hiện cực ít: tea, olives, mango, dried cranberries, apricot, date, salad, fig, tofu, pear, hamburg, watermelon, wonton dumplings...
nhiều object median area rất nhỏ: dried cranberries, blueberry, nut, onion, vegetable, tomato...
foreground trung bình chỉ khoảng 0.49 ảnh, tức background vẫn chiếm mạnh.

So với full dataset, rebalance đúng là đã gộp bớt long-tail cực đoan; ví dụ mushroom, nut, vegetable, seafood được gom lại. Nhưng bản full cũng cho thấy bản chất bài toán vốn đã rất lệch: nhiều class như kelp, enoki mushroom, okra, king oyster mushroom, peanut, egg tart có pixel/presence cực thấp.

Nên kết luận công bằng là:

rebalance là đúng
nhưng rebalance chưa đủ

```
phần còn lại phải xử lý ở mức sampling / crop / split / metric / hard-pair curriculum
```