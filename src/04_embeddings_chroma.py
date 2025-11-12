"""
04_embeddings_chroma.py

- data/docs/ 아래의 NIST PDF 3개를 모두 로드
- 텍스트를 chunk 로 나눈 뒤 OpenAIEmbeddings 로 임베딩 생성
- 로컬 Chroma Vector Store 로 저장(persist)
"""

from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


BASE_DIR = Path(__file__).resolve().parents[1]
DOCS_DIR = BASE_DIR / "data" / "docs"
CHROMA_DIR = BASE_DIR / "chroma_store"


def load_all_pdfs() -> List[Document]:
    pdf_paths = sorted(DOCS_DIR.glob("*.pdf"))
    docs: List[Document] = []

    for path in pdf_paths:
        loader = PyPDFLoader(str(path))
        loaded = loader.load()
        for d in loaded:
            # 간단한 메타데이터: 파일명 추가
            d.metadata.setdefault("source", path.name)
        docs.extend(loaded)

    return docs


def split_documents(docs: List[Document], chunk_size=800, chunk_overlap=120) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)


def build_chroma_index(chunks: List[Document]) -> None:
    CHROMA_DIR.mkdir(exist_ok=True)
    embeddings = OpenAIEmbeddings()

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name="nist-security-docs",
    )
    vectordb.persist()
    print(f"Saved {len(chunks)} chunks into Chroma at {CHROMA_DIR}")


def main() -> None:
    env_path = BASE_DIR / ".env"
    load_dotenv(env_path)

    print("Loading PDFs from:", DOCS_DIR)
    docs = load_all_pdfs()
    print(f"Loaded {len(docs)} pages.")

    chunks = split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")

    build_chroma_index(chunks)


if __name__ == "__main__":
    main()
