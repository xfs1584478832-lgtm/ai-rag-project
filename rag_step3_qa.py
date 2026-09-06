import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()
api_key = os.getenv("SILICONFLOW_API_KEY")

# ========== 1. 初始化大模型客户端 ==========
client = OpenAI(
    api_key=api_key,
    base_url="https://api.siliconflow.cn/v1"
)

# ========== 2. 加载向量数据库（直接用之前存好的，不用重新建） ==========
embeddings = HuggingFaceEmbeddings(
    model_name="shibing624/text2vec-base-chinese",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# ========== 3. RAG 问答函数 ==========
def rag_question(question, top_k=2):
    # 第一步：检索相关文档片段
    docs = db.similarity_search(question, k=top_k)
    
    # 把检索到的片段拼接成一段参考资料
    context = "\n\n".join([doc.page_content for doc in docs])
    
    print("=" * 60)
    print(f"【用户问题】{question}")
    print("=" * 60)
    print(f"【检索到的参考资料】")
    print(context)
    print("=" * 60)
    
    # 第二步：组装 Prompt，告诉大模型基于资料回答
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

    # 第三步：调用大模型生成回答
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3",
        messages=[
            {"role": "system", "content": "你是一个严谨的知识库问答助手，只基于提供的资料回答问题。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,  # 问答场景调低temperature，答案更稳定准确
        max_tokens=1000
    )
    
    answer = response.choices[0].message.content
    print(f"【AI 回答】")
    print(answer)
    print("=" * 60)
    
    return answer

# ========== 4. 测试 ==========
if __name__ == "__main__":
    # 测试1：文档里有的问题
    rag_question("什么是RAG？它有哪些优势？")
    
    print("\n\n")
    
    # 测试2：文档里没有的问题（看AI会不会说"无法回答"）
    rag_question("请介绍一下Python的历史。")
