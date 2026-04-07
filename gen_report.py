import pandas as pd
from pathlib import Path

docs_dir = Path(r"f:\ANHTHU\1-HCMUS\1 - STUDY\HKVIII\CV\PROJECT\source\dataset\foodseg103_rebalanced\docs")
seaweed = pd.read_csv(docs_dir / 'seaweed_check.csv')
missing_train = pd.read_csv(docs_dir / 'classes_missing_in_train.csv')
missing_val = pd.read_csv(docs_dir / 'split_class_coverage.csv')
missing_val = missing_val[missing_val['flags'].str.contains('missing_in_val', na=False)]
rare = pd.read_csv(docs_dir / 'rare_classes_train.csv')

with open(docs_dir / 'report.md', 'w', encoding="utf-8") as f:
    f.write('# Báo cáo Kiểm kê Dataset FoodSeg103 Rebalanced\n\n')
    
    f.write('## 1. Xử lý Dead Class (seaweed)\n')
    f.write(seaweed.to_markdown(index=False) + '\n\n')
    f.write('**Kết luận:** Class seaweed đã bị drop khỏi ontology. Annotations json không chứa class này, label mapping cũng không sinh ra mask cho class này. Mask chỉ trả về 0. Đề xuất drop khỏi ontology trong lúc train để không bị NaN metric.\n\n')

    f.write('## 2. Audit Split Train/Val\n')
    f.write('### Missing in Train\n')
    f.write(missing_train.to_markdown(index=False) + '\n\n')
    f.write('### Missing in Val\n')
    f.write(missing_val.to_markdown(index=False) + '\n\n')
    f.write('### Rare / Ultra Rare in Train\n')
    f.write(rare.to_markdown(index=False) + '\n\n')
    
    f.write('## 3. Rà Patch Lỗi (Texture Analysis)\n')
    patches = pd.read_csv(docs_dir / 'texture_patches_quality.csv')
    f.write(f'Tổng số patch được quét (từ 500 ảnh đầu): {len(patches)}. Số patch có cờ cảnh báo: {len(patches[patches.quality_flag != "ok"])}.\n\n')
    f.write('Dưới đây là một số patch cảnh báo (có thể xem chi tiết trong file `texture_patches_quality.csv` và thư mục `patches`):\n\n')
    f.write(patches[patches.quality_flag != 'ok'].head(20).to_markdown(index=False) + '\n\n')
    
    f.write('## Tổng Hợp Đề Xuất (Actionable Report)\n')
    f.write('| issue_type | item | evidence | action |\n')
    f.write('| --- | --- | --- | --- |\n')
    
    if seaweed.loc[0, 'presence_count'] == 0:
        f.write('| dead_class | seaweed | 0 pixels, 0 presence | Drop khỏi ontology train/test để tránh NaN |\n')
    
    if len(missing_train) > 0:
        names = ', '.join(missing_train['class_name'].tolist())
        f.write(f'| split_leak | {names} | có ở test nhưng không có ở train | Xem lại split hoặc move sample về train |\n')
        
    if len(missing_val) > 0:
        names = ', '.join(missing_val['class_name'].tolist())
        f.write(f'| split_leak | {names} | có ở train nhưng thiếu val | Cần sample thêm vào val/test set |\n')
        
    if len(rare) > 0:
        f.write(f'| data_starved | ultra_rare/rare classes ({len(rare)}) | presence_count <= 5 | Data augmentation hoặc merge class |\n')
        
    f.write('| patch_quality | Hard pairs patches | nhiều patches bị quá mờ (blurry), hoặc crop quá chặt (too_tight) | Nới rộng margin khi crop patch (+10%), áp dụng deblur/augment |\n')

print("Report saved successfully.")
