"""
tools/data_processing/labelsTOyolo.py
用途：将 YOLO 标注中所有类别 ID 统一重置为 0（用于单类检测训练前的标签预处理）
说明：
- 输入：原始标注目录（labels_raw），每个 txt 文件每行格式为 class_id x y w h
- 输出：新标注目录（labels），所有 class_id 强制写为 0
- 运行方式：直接改下面 2 个路径，然后 python tools/labelsTOyolo.py
"""

from pathlib import Path

# ========== 改这里 ==========
src_dir = Path("labels_raw")           # 改为实际原始标注目录路径
dst_dir = Path("labels")               # 改为实际输出标注目录路径
# ============================

dst_dir.mkdir(exist_ok=True)

for txt_file in src_dir.glob("*.txt"):
    new_lines = []
    with open(txt_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                parts[0] = "0"
                new_lines.append(" ".join(parts))

    with open(dst_dir / txt_file.name, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines))

print("✅ 转换完成，类别已全部设为 0")
print(f"   输入: {src_dir}")
print(f"   输出: {dst_dir}")