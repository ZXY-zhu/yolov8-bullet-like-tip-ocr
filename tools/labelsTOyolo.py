# tools/labelsTOyolo.py
# 用途：统一 YOLO 标签类别 ID 为 0，修复 LabelImg 映射问题

from pathlib import Path

src_dir = Path("C:/Users/BeLig/Desktop/ocr/no/labels")        # 原来的标签
dst_dir = Path("C:/Users/BeLig/Desktop/ocr/no/labels_yolo")   # 新文件夹

dst_dir.mkdir(exist_ok=True)  # 如果文件夹不存在就创建

for txt_file in src_dir.glob("*.txt"):
    new_lines = []
    with open(txt_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                parts[0] = "0"  # 只改类别
                new_lines.append(" ".join(parts))

    # 写到新文件夹，文件名一样
    with open(dst_dir / txt_file.name, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines))

print("✅ 转换完成，原始 labels 未被修改")