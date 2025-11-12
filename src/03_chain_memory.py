"""
03_chain_memory.py

- LLMChain / SimpleSequentialChain 사용 예시
- ConversationBufferMemory 를 활용한 간단 대화형 챗봇
"""

from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory


def run_sequential_chain() -> None:
    """두 개의 LLMChain을 순차적으로 연결하는 예제."""
    llm = ChatOpenAI(model="gpt-4o-mini")

    summ_prompt = PromptTemplate.from_template(
        "다음 내용을 한 문장으로 요약해줘:\n\n{text}"
    )
    style_prompt = PromptTemplate.from_template(
        "다음을 더 친근하고 이해하기 쉬운 톤으로 바꿔줘:\n\n{text}"
    )

    # 1단계: 요약 → string 출력
    summ_chain = summ_prompt | llm | StrOutputParser()

    # string → {"text": string} 로 변환해서 다음 프롬프트에 넣기
    to_style_input = RunnableLambda(lambda s: {"text": s})

    # 2단계: 톤 변경 → string 출력
    style_chain = style_prompt | llm | StrOutputParser()

    # 전체 파이프: 요약 → (형 변환) → 톤 변경
    overall_chain = summ_chain | to_style_input | style_chain

    text = (
        "LangChain과 LangSmith를 활용하면 LLM 기반 애플리케이션의 구성, "
        "디버깅, 모니터링을 훨씬 더 쉽게 할 수 있습니다."
    )

    result = overall_chain.invoke({"text": text})  # 또는 overall_chain.invoke(text)도 동작
    print(result)


def run_conversation_chain() -> None:
    """Memory를 활용한 간단 대화형 챗봇 예제."""
    # llm 이미 있으시면 재사용하세요. 없으면 주석 해제:
    llm = ChatOpenAI(model="gpt-4o-mini")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "너는 GC케어 LangChain 교육의 친절한 조교야. 간결하고 명확히 답해줘."),
        MessagesPlaceholder("history"),  # ← 과거 대화를 여기에 주입
        ("human", "{input}")
    ])

    chain = prompt | llm  # LCEL

    # 세션별 히스토리 저장소 (실무에선 Redis/DB 등으로 대체)
    session_store = {}

    def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in session_store:
            session_store[session_id] = InMemoryChatMessageHistory()
        return session_store[session_id]

    chat_with_memory = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",   # 프롬프트에서 사람 입력 변수명
        history_messages_key="history" # MessagesPlaceholder 키
    )

    # === 사용 예시 ===
    session_id = "demo-user-1"

    print('--- 첫 번째 질문 ---')
    res1 = chat_with_memory.invoke(
        {"input": "오늘 GC케어 LangChain 교육의 목표를 한 문장으로 말해줘."},
        config={"configurable": {"session_id": session_id}},
    )
    print(res1.content)

    print('\n--- 두 번째 질문 (이전 답변을 참고해야 함) ---')
    res2 = chat_with_memory.invoke(
        {"input": "방금 말한 내용을 두 줄로 다시 정리해줘."},
        config={"configurable": {"session_id": session_id}},
    )
    print('\n최종 응답:', res2.content)


def main() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(env_path)

    run_sequential_chain()
    run_conversation_chain()


if __name__ == "__main__":
    main()
