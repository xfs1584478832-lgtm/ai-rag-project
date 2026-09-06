from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. 加载并切块（和上一步一样）
loader = TextLoader("test_doc.txt", encoding="utf-8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
)
chunks = text_splitter.split_documents(documents)
print(f"切成了 {len(chunks)} 个块")

# 2. 初始化 embedding 模型（把文字转成向量）
# 用的是一个开源的中文 embedding 模型，第一次运行会自动下载（约几百MB）
print("正在加载 embedding 模型，第一次需要下载，请稍候...")
embeddings = HuggingFaceEmbeddings(
    model_name="shibing624/text2vec-base-chinese",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 3. 把块存进 Chroma 向量数据库
print("正在向量化并存入数据库...")
db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"  # 数据库存在本地这个文件夹里
)

print(f"成功存入 {len(chunks)} 个向量到数据库！")
print("数据库文件已保存到 ./chroma_db 文件夹")

# 4. 测试检索：问一个问题，看能找到相关的块吗？
query = "什么是RAG？它有什么优势？"
print(f"\n测试检索问题：{query}")

results = db.similarity_search(query, k=2)  # 找最相关的2个块

print(f"\n找到 {len(results)} 个相关块：")
print("-" * 50)
for i, doc in enumerate(results):
    print(f"【相关块 {i+1}】")
    print(doc.page_content)
    print("-" * 50)
