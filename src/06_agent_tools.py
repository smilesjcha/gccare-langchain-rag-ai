"""
06_agent_tools.py (LangGraph 기반)

- 계산/시간변환 Tool + ReAct Agent 구성
- LangSmith Trace에서 Agent/Tool 호출 구조 확인
- Hub 없이 state/messages modifier로 안정 실행
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

# ✅ LangGraph (신규 표준)
from langgraph.prebuilt import create_react_agent


BASE_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = BASE_DIR / ".env"

# =========
# Tools
# =========

@tool
def add(a: int, b: int) -> int:
    """두 정수 a와 b의 합을 반환합니다."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """두 정수 a와 b의 곱을 반환합니다."""
    return a * b

@tool
def to_kst(iso_dt: str) -> str:
    """UTC ISO8601 날짜문자열(예:'2025-11-13T06:00:00Z')을 한국시간(KST, UTC+9)으로 변환합니다."""
    from datetime import datetime, timezone, timedelta
    dt = datetime.fromisoformat(iso_dt.replace("Z", "+00:00"))
    kst = dt.astimezone(timezone(timedelta(hours=9)))
    return kst.strftime("%Y-%m-%d %H:%M:%S KST")


SYSTEM_PROMPT = (
    "너는 계산 및 시간 변환 도구를 활용해 단계적으로 문제를 푸는 조교다. "
    "필요하면 도구를 사용하고, 최종적으로 간결한 한국어 답변을 제시하라. "
    "도구 입력은 JSON 형식을 유지하라."
)


def build_agent():
    """
    LangGraph의 create_react_agent 버전 차이를 자동 호환:
    - v2: state_modifier, version="v2"
    - v1: messages_modifier, version="v1"
    - fallback: modifier 없이 생성 후 호출 시 SystemMessage로 주입
    """
    llm = ChatOpenAI(model="gpt-4o-mini")
    tools = [add, multiply, to_kst]

    # v2 우선
    try:
        return create_react_agent(
            model=llm,
            tools=tools,
            state_modifier=SYSTEM_PROMPT,  # v2 계열
            version="v2",
        )
    except TypeError:
        pass

    # v1 시도
    try:
        return create_react_agent(
            model=llm,
            tools=tools,
            messages_modifier=SYSTEM_PROMPT,  # v1 계열
            version="v1",
        )
    except TypeError:
        pass

    # 마지막 폴백: modifier 없이라도 생성
    return create_react_agent(model=llm, tools=tools)


def run_stateless(agent, questions: List[str]) -> None:
    """각 질문을 독립 실행(세션 유지 없음)."""
    for q in questions:
        print("\n==============================")
        print("질문:", q)
        try:
            res = agent.invoke(
                {"messages": [HumanMessage(content=q)]},
                config={"recursion_limit": 8},
            )
        except Exception:
            # modifier 없이 생성된 폴백 에이전트 대비
            res = agent.invoke(
                {"messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=q)]},
                config={"recursion_limit": 8},
            )
        msgs = res.get("messages", [])
        answer = msgs[-1].content if msgs else str(res)
        print("\n최종 답변:", answer)


def run_stateful(agent, questions: List[str]) -> None:
    """동일 세션에서 연속 질문(대화 히스토리 유지)."""
    history: List = []
    for q in questions:
        print("\n==============================")
        print("질문:", q)
        history.append(HumanMessage(content=q))
        try:
            out = agent.invoke({"messages": history}, config={"recursion_limit": 8})
        except Exception:
            # modifier 없이 생성된 폴백 에이전트 대비
            out = agent.invoke(
                {"messages": [SystemMessage(content=SYSTEM_PROMPT), *history]},
                config={"recursion_limit": 8},
            )
        history[:] = out["messages"]
        print("\n최종 답변:", history[-1].content)


def main() -> None:
    # .env 로드 (OpenAI/LangSmith 키 등)
    load_dotenv(ENV_PATH)

    agent = build_agent()

    print("=== Stateless 실행 ===")
    run_stateless(
        agent,
        [
            "10과 32를 더한 다음, 그 결과를 2배로 만든 값을 알려줘.",
            "'2025-11-13T06:00:00Z' 를 한국시간으로 바꿔줘.",
            "7과 8을 곱한 뒤, 10을 더해줘. 필요한 계산은 도구를 사용해.",
        ],
    )

    print("\n\n=== Stateful 실행 (대화형) ===")
    run_stateful(
        agent,
        [
            "12와 30을 더해줘.",
            "방금 결과를 한 문장으로 설명해줘.",
        ],
    )


if __name__ == "__main__":
    main()
