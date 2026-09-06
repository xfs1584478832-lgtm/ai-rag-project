import os
import shutil
from dotenv import load_dotenv
from openai import OpenAI
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# ========== 初始化 ==========
app = FastAPI(title="AI 知识库问答系统")

# 允许跨域（前端页面调用后端接口需要）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 大模型客户端
client = OpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1"
)

# Embedding 模型
embeddings = HuggingFaceEmbeddings(
    model_name="shibing624/text2vec-base-chinese",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 向量数据库路径
CHROMA_PATH = "./chroma_db"
UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 文本切块器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
)


def get_vector_db():
    """获取或创建向量数据库"""
    if os.path.exists(CHROMA_PATH):
        return Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    return None


# ========== 接口 1：首页 ==========
@app.get("/")
async def root():
    return {"message": "AI 知识库问答系统 API 已启动", "docs": "/docs"}


# ========== 接口 2：上传文档，建立知识库 ==========
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """上传 TXT 或 PDF 文件，自动解析并存入向量数据库"""
    
    # 保存上传的文件
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # 根据文件类型选择加载器
    if file.filename.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    elif file.filename.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        raise HTTPException(status_code=400, detail="仅支持 TXT 和 PDF 格式")
    
    # 加载并切块
    documents = loader.load()
    chunks = text_splitter.split_documents(documents)
    
    # 存入向量数据库
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    
    return {
        "message": "文档上传成功",
        "filename": file.filename,
        "chunks_count": len(chunks)
    }


# ========== 接口 3：提问 ==========
@app.get("/ask")
async def ask_question(question: str, top_k: int = 2):
    """基于知识库回答问题"""
    
    db = get_vector_db()
    if db is None:
        raise HTTPException(status_code=400, detail="知识库为空，请先上传文档")
    
    # 检索相关文档
    docs = db.similarity_search(question, k=top_k)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # 组装 Prompt
    prompt = f"""你是一个知识库问答助手。请严格根据下面提供的参考资料回答用户的问题。

要求：
1. 只使用参考资料中的信息，不要自己编造
2. 如果参考资料中没有答案，请回答"根据现有资料，无法回答这个问题"
3. 回答简洁明了，分点说明

【参考资料】
{context}

【用户问题】
{question}

【你的回答】"""
    
    # 调用大模型
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3",
        messages=[
            {"role": "system", "content": "你是一个严谨的知识库问答助手，只基于提供的资料回答问题。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=1000
    )
    
    answer = response.choices[0].message.content
    
    return {
        "question": question,
        "answer": answer,
        "sources": [doc.page_content[:100] + "..." for doc in docs]  # 返回引用来源
    }


# ========== 接口 4：清空知识库 ==========
@app.delete("/clear")
async def clear_knowledge_base():
    """清空当前知识库"""
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    return {"message": "知识库已清空"}
