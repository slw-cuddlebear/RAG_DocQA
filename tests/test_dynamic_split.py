from dotenv import load_dotenv
load_dotenv()
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_core import load_local_document
from dynamic_split import dynamic_split


def on_progress(p, msg):
    print(f"[{p:3d}%] {msg}")


# 加载文档
docs = load_local_document("data/bp.docx")
print(f"原始文档：{len(docs)} 页，共 {sum(len(d.page_content) for d in docs)} 字符\n")

# 动态切片
chunks = dynamic_split(docs, progress_callback=on_progress)

# 查看结果
print(f"\n===== 共 {len(chunks)} 个切片 =====")
for i, c in enumerate(chunks[:5], 1):
    print(f"\n--- 切片 {i}（{len(c.page_content)} 字）---")
    print(c.page_content[:200])