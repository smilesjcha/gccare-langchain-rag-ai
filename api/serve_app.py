from __future__ import annotations

"""
serve_app.py
- FastAPI + LangServe 서버
- 엔드포인트:
  1) /rag   : NIST PDF 기반 RAG (Chroma + OpenAIEmbeddings + LCEL)

필수 설치:
  pip install -U fastapi uvicorn langserve langchain-core langchain-openai \
                 langchain-community langchain-text-splitters chromadb python-dotenv

환경 변수(.env):
  OPENAI_API_KEY=sk-...
  LANGCHAIN_TRACING_V2=true
  LANGCHAIN_PROJECT=gccare-rag-workshop
  LANGCHAIN_API_KEY=ls-...
"""

import os
from glob import glob
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from langserve import add_routes

# ===== RAG(Chroma + LCEL) =====
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# ===========================
# 공통 설정
# ===========================
load_dotenv()  # .env 로드
APP_TITLE = "GCcare NIST RAG API"
APP_VERSION = "1.2.0"

# 현재 파일 기준 데이터 경로
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data" / "docs"
PERSIST_DIR = HERE / "chroma_db"
PERSIST_DIR.mkdir(parents=True, exist_ok=True)


# ===========================
# RAG (Chroma + LCEL)
# ===========================
def load_all_pdfs(data_dir: Path):
    paths = sorted(glob(str(data_dir / "*.pdf")))
    docs = []
    for p in paths:
        try:
            loader = PyPDFLoader(p)
            docs.extend(loader.load())
        except Exception as e:
            print(f"[WARN] PDF 로드 실패: {p} -> {e}")
    return docs


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def build_rag_chain():
    # 0) 임베딩 먼저 준비
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 1) 문서 로드
    raw_docs = load_all_pdfs(DATA_DIR)

    if raw_docs:
        # PDF가 있을 때만 벡터스토어 재생성
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
        splits = splitter.split_documents(raw_docs)

        if splits:
            vectordb = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=str(PERSIST_DIR),
            )
        else:
            # 이론상 거의 없겠지만, 혹시라도 splits가 비면 여기로
            print("[WARN] 청크가 비어 있습니다. 기존 벡터스토어만 로드합니다.")
            vectordb = Chroma(
                embedding_function=embeddings,
                persist_directory=str(PERSIST_DIR),
            )
    else:
        # 📌 지금 상황: PDF가 하나도 없을 때
        print(f"[WARN] PDF가 없습니다: {DATA_DIR}. 빈 벡터스토어를 로드합니다.")
        vectordb = Chroma(
            embedding_function=embeddings,
            persist_directory=str(PERSIST_DIR),
        )

    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    # 3) 프롬프트 & LCEL 체인
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        주어진 컨텍스트를 사용해서 사용자의 질문에 한국어로 정확하고 간결하게 답하라.
        컨텍스트에 없으면 모른다고 답하라.

        [컨텍스트]
        {context}
        """.strip()),
        ("human", "질문: {question}")
    ])

    llm = ChatOpenAI(model="gpt-4o-mini")

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


# ===========================
# FastAPI + LangServe
# ===========================
app = FastAPI(title=APP_TITLE, version=APP_VERSION)


@app.get("/health")
def health():
    return {"status": "ok"}


# RAG 마운트 (입력: {"input": "질문 문자열"})
rag_chain = build_rag_chain()
add_routes(app, rag_chain, path="/rag")

print("App ready: /health, /rag")
