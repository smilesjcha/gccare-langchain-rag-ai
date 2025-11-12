"""
02_prompt_basics.py

- ChatPromptTemplate + ChatOpenAI 기본 사용법
- LCEL 파이프라인(pipeline-style) 체인 구성
"""

from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def build_chain():
    """프롬프트 + LLM으로 구성된 가장 기본적인 체인을 만든다."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "너는 GC케어 개발자 교육을 돕는 친절한 LangChain 조교야.",
            ),
            (
                "user",
                "다음 텍스트를 3줄로 요약해줘.\n\n{text}",
            ),
        ]
    )
    llm = ChatOpenAI(model="gpt-4o-mini")
    return prompt | llm


def main() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(env_path)

    chain = build_chain()

    sample_text = (
        "이번 교육에서는 LangChain과 LangSmith, Chroma를 활용해 "
        "PDF 기반 RAG 파이프라인을 직접 구현하고, "
        "FastAPI 및 LangServe로 API까지 배포하는 과정을 실습합니다."
    )

    res = chain.invoke({"text": sample_text})
    print(res.content)


if __name__ == "__main__":
    main()
