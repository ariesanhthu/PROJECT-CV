# FoodSeg103 Rebalanced Dataset Description

## Overview

- Dataset path: `source/dataset/foodseg103_rebalanced`
- Source reference used for audit: `source/dataset/foodseg103-full`
- Total classes after rebalance: **75**
- Total images: **6116** (`train=3981`, `test=2135`)
- Total annotations: **6116**
- Total labeled objects: **22650**
- Average objects per image: **3.70**
- Total foreground pixels across all decoded masks: **2,251,537,230**

## Integrity Check

- Checked all `6116/6116` images with Pillow: **no corrupted image file found**.
- Checked all `6116/6116` annotation JSON files: **all JSON files are readable**.
- Checked all `22650/22650` bitmap masks: **all masks can be decoded successfully**.
- File pairing is complete: `missing_ann_for_img = 0`, `missing_img_for_ann = 0` for both `train` and `test`.
- Detected **3** annotation size mismatches where JSON `size` is swapped relative to real image size:
  - `train/img/00000273.jpg`: real size `(384, 512)`, JSON size `(512, 384)`
  - `train/img/00002585.jpg`: real size `(384, 512)`, JSON size `(512, 384)`
  - `train/img/00003969.jpg`: real size `(384, 512)`, JSON size `(512, 384)`
- Detected **1** out-of-bounds bitmap mask inherited from the source annotation:
  - `train/00003969.jpg`, object id `150556539`, class `pork`, origin `(74, 234)`, mask size `(311, 150)`, image size `(384, 512)`

## Label Audit

- Compared every rebalanced object against the corresponding object in `foodseg103-full` using `object id`, `bitmap origin`, `bitmap payload`, and `class_mapping.csv`.
- Result: **no remap error found** (`remap = {}`).
- Result: **no class-id/title mismatch found** in the rebalanced metadata.
- Result: **all 75 classes have at least one object**; there is no empty class after rebalance.
- Structural conclusion: there is **no evidence of wrong class reassignment introduced by the rebalance step**.
- Note: this audit validates annotation integrity and remapping consistency. It does **not** guarantee semantic correctness by visual inspection for every image.

## Distribution Summary

Top 10 classes by object count:
- 1. `bread`: objects=1259, images=1259, object_share=5.56%, pixel_share=7.46%
- 2. `carrot`: objects=1172, images=1172, object_share=5.17%, pixel_share=2.31%
- 3. `chicken duck`: objects=1124, images=1124, object_share=4.96%, pixel_share=10.81%
- 4. `sauce`: objects=1051, images=1051, object_share=4.64%, pixel_share=3.00%
- 5. `tomato`: objects=1034, images=1034, object_share=4.57%, pixel_share=2.10%
- 6. `potato`: objects=1001, images=1001, object_share=4.42%, pixel_share=5.50%
- 7. `steak`: objects=983, images=983, object_share=4.34%, pixel_share=5.19%
- 8. `broccoli`: objects=935, images=935, object_share=4.13%, pixel_share=4.20%
- 9. `ice cream`: objects=790, images=790, object_share=3.49%, pixel_share=2.23%
- 10. `cilantro mint`: objects=769, images=769, object_share=3.40%, pixel_share=1.17%

Top 10 classes by pixel share:
- 1. `chicken duck`: pixel_share=10.81%, objects=1124, images=1124
- 2. `bread`: pixel_share=7.46%, objects=1259, images=1259
- 3. `potato`: pixel_share=5.50%, objects=1001, images=1001
- 4. `pork`: pixel_share=5.39%, objects=589, images=589
- 5. `steak`: pixel_share=5.19%, objects=983, images=983
- 6. `rice`: pixel_share=4.61%, objects=611, images=611
- 7. `broccoli`: pixel_share=4.20%, objects=935, images=935
- 8. `pie`: pixel_share=3.12%, objects=494, images=494
- 9. `corn`: pixel_share=3.08%, objects=454, images=454
- 10. `sauce`: pixel_share=3.00%, objects=1051, images=1051

10 rarest classes by object count:
- 1. `hamburg`: objects=6, images=6, object_share=0.0265%
- 2. `seaweed`: objects=7, images=7, object_share=0.0309%
- 3. `apricot`: objects=10, images=10, object_share=0.0442%
- 4. `fig`: objects=10, images=10, object_share=0.0442%
- 5. `crab`: objects=13, images=13, object_share=0.0574%
- 6. `watermelon`: objects=16, images=16, object_share=0.0706%
- 7. `wonton dumplings`: objects=16, images=16, object_share=0.0706%
- 8. `eggplant`: objects=17, images=17, object_share=0.0751%
- 9. `dried cranberries`: objects=18, images=18, object_share=0.0795%
- 10. `salad`: objects=20, images=20, object_share=0.0883%

## Tag Consistency After Merge

- `beans_and_peas` has mixed tags: `vegetable=612`, `nut=63`. This is expected from merged source classes, not a remap failure.
- `other ingredients` has mixed tags: `other ingredients=198`, `fruit=5`, `vegetable=3`. This is also expected from merged source classes.
- No other class shows mixed tag structure after rebalance.

## Full Class Distribution

| ID | Class | Objects | Images | Object Share | Pixel Share | Train Obj | Test Obj | Train Img | Test Img |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `alliums_and_garlic` | 700 | 660 | 3.0905% | 1.2575% | 457 | 243 | 424 | 236 |
| 2 | `apple` | 134 | 134 | 0.5916% | 0.2796% | 82 | 52 | 82 | 52 |
| 3 | `apricot` | 10 | 10 | 0.0442% | 0.0293% | 2 | 8 | 2 | 8 |
| 4 | `asparagus` | 233 | 233 | 1.0287% | 0.8274% | 156 | 77 | 156 | 77 |
| 5 | `avocado` | 66 | 66 | 0.2914% | 0.2672% | 42 | 24 | 42 | 24 |
| 6 | `banana` | 136 | 136 | 0.6004% | 0.3921% | 80 | 56 | 80 | 56 |
| 7 | `beans_and_peas` | 675 | 661 | 2.9801% | 2.5379% | 424 | 251 | 417 | 244 |
| 8 | `blueberry` | 190 | 190 | 0.8389% | 0.3183% | 132 | 58 | 132 | 58 |
| 9 | `bread` | 1259 | 1259 | 5.5585% | 7.4586% | 845 | 414 | 845 | 414 |
| 10 | `broccoli` | 935 | 935 | 4.1280% | 4.2036% | 626 | 309 | 626 | 309 |
| 11 | `carrot` | 1172 | 1172 | 5.1744% | 2.3125% | 774 | 398 | 774 | 398 |
| 12 | `cauliflower` | 213 | 213 | 0.9404% | 0.5352% | 143 | 70 | 143 | 70 |
| 13 | `celery stick` | 209 | 209 | 0.9227% | 0.4510% | 144 | 65 | 144 | 65 |
| 14 | `cheese butter` | 333 | 333 | 1.4702% | 1.2146% | 213 | 120 | 213 | 120 |
| 15 | `cherry` | 183 | 183 | 0.8079% | 0.3071% | 130 | 53 | 130 | 53 |
| 16 | `chicken duck` | 1124 | 1124 | 4.9625% | 10.8125% | 730 | 394 | 730 | 394 |
| 17 | `cilantro mint` | 769 | 769 | 3.3951% | 1.1721% | 505 | 264 | 505 | 264 |
| 18 | `citrus` | 738 | 724 | 3.2583% | 1.2825% | 496 | 242 | 487 | 237 |
| 19 | `coffee` | 156 | 156 | 0.6887% | 0.1301% | 99 | 57 | 99 | 57 |
| 20 | `corn` | 454 | 454 | 2.0044% | 3.0849% | 309 | 145 | 309 | 145 |
| 21 | `crab` | 13 | 13 | 0.0574% | 0.1236% | 7 | 6 | 7 | 6 |
| 22 | `cucumber` | 492 | 492 | 2.1722% | 1.7820% | 328 | 164 | 328 | 164 |
| 23 | `desserts` | 697 | 652 | 3.0773% | 2.7895% | 459 | 238 | 432 | 220 |
| 24 | `dried cranberries` | 18 | 18 | 0.0795% | 0.0220% | 6 | 12 | 6 | 12 |
| 25 | `egg` | 258 | 258 | 1.1391% | 1.4618% | 189 | 69 | 189 | 69 |
| 26 | `eggplant` | 17 | 17 | 0.0751% | 0.0575% | 10 | 7 | 10 | 7 |
| 27 | `fig` | 10 | 10 | 0.0442% | 0.0329% | 5 | 5 | 5 | 5 |
| 28 | `fish` | 316 | 316 | 1.3951% | 2.3042% | 221 | 95 | 221 | 95 |
| 29 | `french fries` | 222 | 222 | 0.9801% | 1.7370% | 144 | 78 | 144 | 78 |
| 30 | `fried meat` | 222 | 222 | 0.9801% | 1.8346% | 138 | 84 | 138 | 84 |
| 31 | `grape` | 144 | 144 | 0.6358% | 0.3016% | 109 | 35 | 109 | 35 |
| 32 | `hamburg` | 6 | 6 | 0.0265% | 0.0565% | 5 | 1 | 5 | 1 |
| 33 | `hanamaki baozi` | 27 | 27 | 0.1192% | 0.1253% | 18 | 9 | 18 | 9 |
| 34 | `ice cream` | 790 | 790 | 3.4879% | 2.2328% | 513 | 277 | 513 | 277 |
| 35 | `juice` | 146 | 146 | 0.6446% | 0.5137% | 92 | 54 | 92 | 54 |
| 36 | `kiwi` | 45 | 45 | 0.1987% | 0.1019% | 32 | 13 | 32 | 13 |
| 37 | `lamb` | 85 | 85 | 0.3753% | 0.5169% | 57 | 28 | 57 | 28 |
| 38 | `leafy_greens` | 694 | 679 | 3.0640% | 2.6072% | 469 | 225 | 457 | 222 |
| 39 | `mango` | 30 | 30 | 0.1325% | 0.0516% | 18 | 12 | 18 | 12 |
| 40 | `melon` | 21 | 21 | 0.0927% | 0.0929% | 17 | 4 | 17 | 4 |
| 41 | `milk` | 53 | 53 | 0.2340% | 0.1169% | 25 | 28 | 25 | 28 |
| 42 | `milkshake` | 81 | 81 | 0.3576% | 0.3613% | 54 | 27 | 54 | 27 |
| 43 | `mushroom` | 238 | 234 | 1.0508% | 0.6299% | 139 | 99 | 138 | 96 |
| 44 | `noodles` | 217 | 217 | 0.9581% | 2.6827% | 142 | 75 | 142 | 75 |
| 45 | `nut` | 94 | 92 | 0.4150% | 0.0787% | 36 | 58 | 35 | 57 |
| 46 | `olives` | 43 | 43 | 0.1898% | 0.0363% | 28 | 15 | 28 | 15 |
| 47 | `other ingredients` | 206 | 206 | 0.9095% | 0.5969% | 129 | 77 | 129 | 77 |
| 48 | `pasta` | 147 | 147 | 0.6490% | 2.4500% | 99 | 48 | 99 | 48 |
| 49 | `peach` | 32 | 32 | 0.1413% | 0.1148% | 14 | 18 | 14 | 18 |
| 50 | `pear` | 28 | 28 | 0.1236% | 0.0581% | 14 | 14 | 14 | 14 |
| 51 | `pepper` | 381 | 381 | 1.6821% | 0.9827% | 250 | 131 | 250 | 131 |
| 52 | `pie` | 494 | 494 | 2.1810% | 3.1152% | 333 | 161 | 333 | 161 |
| 53 | `pineapple` | 67 | 67 | 0.2958% | 0.2399% | 31 | 36 | 31 | 36 |
| 54 | `pizza` | 56 | 56 | 0.2472% | 0.3708% | 43 | 13 | 43 | 13 |
| 55 | `pork` | 589 | 589 | 2.6004% | 5.3898% | 394 | 195 | 394 | 195 |
| 56 | `potato` | 1001 | 1001 | 4.4194% | 5.5033% | 695 | 306 | 695 | 306 |
| 57 | `pumpkin` | 47 | 47 | 0.2075% | 0.2158% | 31 | 16 | 31 | 16 |
| 58 | `raspberry` | 64 | 64 | 0.2826% | 0.1203% | 42 | 22 | 42 | 22 |
| 59 | `rice` | 611 | 611 | 2.6976% | 4.6143% | 405 | 206 | 405 | 206 |
| 60 | `salad` | 20 | 20 | 0.0883% | 0.0339% | 10 | 10 | 10 | 10 |
| 61 | `sauce` | 1051 | 1051 | 4.6402% | 2.9981% | 724 | 327 | 724 | 327 |
| 62 | `sausage` | 205 | 205 | 0.9051% | 1.2239% | 152 | 53 | 152 | 53 |
| 63 | `seaweed` | 7 | 7 | 0.0309% | 0.0220% | 0 | 7 | 0 | 7 |
| 64 | `shellfish` | 24 | 24 | 0.1060% | 0.1684% | 11 | 13 | 11 | 13 |
| 65 | `shrimp` | 116 | 116 | 0.5121% | 0.7444% | 71 | 45 | 71 | 45 |
| 66 | `soup` | 79 | 79 | 0.3488% | 0.3406% | 57 | 22 | 57 | 22 |
| 67 | `steak` | 983 | 983 | 4.3400% | 5.1873% | 646 | 337 | 646 | 337 |
| 68 | `strawberry` | 514 | 514 | 2.2693% | 1.2837% | 342 | 172 | 342 | 172 |
| 69 | `tea` | 23 | 23 | 0.1015% | 0.0178% | 17 | 6 | 17 | 6 |
| 70 | `tofu` | 27 | 27 | 0.1192% | 0.0772% | 7 | 20 | 7 | 20 |
| 71 | `tomato` | 1034 | 1034 | 4.5651% | 2.1041% | 685 | 349 | 685 | 349 |
| 72 | `watermelon` | 16 | 16 | 0.0706% | 0.0713% | 8 | 8 | 8 | 8 |
| 73 | `white radish` | 40 | 40 | 0.1766% | 0.0919% | 24 | 16 | 24 | 16 |
| 74 | `wine` | 104 | 104 | 0.4592% | 0.2172% | 63 | 41 | 63 | 41 |
| 75 | `wonton dumplings` | 16 | 16 | 0.0706% | 0.1207% | 6 | 10 | 6 | 10 |
