from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. 加载文档
loader = TextLoader("test_doc.txt", encoding="utf-8")
documents = loader.load()

print(f"加载了 {len(documents)} 个文档")
print(f"文档总字数：{len(documents[0].page_content)}")
print("-" * 50)

# 2. 切块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,      # 每块最多多少字
    chunk_overlap=50,    # 块与块之间重叠多少字（防止切断语义）
    separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
)

chunks = text_splitter.split_documents(documents)

# 3. 打印结果
print(f"切成了 {len(chunks)} 个块")
print("-" * 50)

for i, chunk in enumerate(chunks):
    print(f"【第 {i+1} 块】（{len(chunk.page_content)} 字）")
    print(chunk.page_content)
    print("-" * 50)
