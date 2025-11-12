"""
01_langsmith_setup.py

- .env 로드
- OpenAI / LangSmith 환경변수 정상 세팅 여부 확인
- 가장 간단한 LCEL 체인을 한 번 실행해서 Trace가 잘 찍히는지 확인
"""

from pathlib import Path
import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def main() -> None:
    # .env 로드
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()  # 이미 로드되어 있을 수도 있음

    print("=== Environment check ===")
    print("OPENAI_API_KEY set?:", bool(os.getenv("OPENAI_API_KEY")))
    print("LANGCHAIN_API_KEY(LangSmith) set?:", bool(os.getenv("LANGCHAIN_API_KEY")))
    print("LANGCHAIN_TRACING_V2:", os.getenv("LANGCHAIN_TRACING_V2"))
    print("LANGCHAIN_PROJECT:", os.getenv("LANGCHAIN_PROJECT"))
    print()

    # 가장 단순한 체인
    prompt = ChatPromptTemplate.from_template(
        "오늘 GC케어 LangChain/RAG 강의의 핵심 목표를 한 문장으로 설명해줘."
    )
    llm = ChatOpenAI(model="gpt-4o-mini")

    chain = prompt | llm

    print("=== Invoking chain (this should appear in LangSmith traces) ===")
    res = chain.invoke({})
    print(res.content)


if __name__ == "__main__":
    main()
