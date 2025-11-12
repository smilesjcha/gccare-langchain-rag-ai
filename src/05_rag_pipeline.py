"""
rag_pipeline.py

- 04_embeddings_chroma.py 에서 생성한 Chroma 인덱스를 로드
- Retriever + LCEL 기반 RAG 체인 구성
- 강의 전체에서 재사용할 rag_chain 객체를 제공
"""

from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


BASE_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)
CHROMA_DIR = BASE_DIR / "chroma_store"


def get_retriever():
    """Persisted Chroma 인덱스를 로드해 Retriever를 반환."""
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name="nist-security-docs",
    )
    return vectordb.as_retriever(search_kwargs={"k": 4})


def get_rag_chain():
    """Chroma retriever + Prompt + LLM 으로 구성된 RAG 체인 생성."""
    retriever = get_retriever()

    prompt = ChatPromptTemplate.from_template(
        """
당신은 NIST 보안/AI 프레임워크 문서를 기반으로 답변하는 전문가입니다.
반드시 아래 Context 안의 내용만 사용하여 한국어로 답변하세요.
모르겠거나 문서에 없는 내용이면 솔직하게 모른다고 말하세요.

# Context
{context}

# Question
{question}
""".strip()
    )

    llm = ChatOpenAI(model="gpt-4o-mini")

    rag_chain = (
        RunnableParallel(
            {
                "context": retriever,
                "question": RunnablePassthrough(),
            }
        )
        | prompt
        | llm
    )

    return rag_chain


# 다른 모듈에서 import 해서 쓸 수 있도록 전역 객체로 노출
rag_chain = get_rag_chain()


def main() -> None:
    question = "Zero Trust Architecture의 핵심 원칙을 3가지로 정리해줘."
    res = rag_chain.invoke(question)
    print(res.content)


if __name__ == "__main__":
    main()
