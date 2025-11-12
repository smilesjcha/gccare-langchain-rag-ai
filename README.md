# 🧠 GC케어 LangChain/RAG 기반 LLM 서비스 개발 강의 계획서

- **일시**: 2025-11-13(목), 11-20(목) / 09:30–17:30 (점심 12:00–13:00)
- **대상**: GC케어 · 유비케어 개발자 및 실무진 (개발자 중심)
- **실습 Repo**: `smilesjcha/gccare-langchain-rag-ai`
- **벡터스토어**: Chroma (로컬)
- **Tracing**: LangSmith (LangChain Tracing V2)

---

## 1. 교육 목표

1. LangChain / LangSmith / Chroma의 역할과 관계를 이해한다.
2. NIST 보안/AI 프레임워크 PDF 3종을 기반으로 RAG 파이프라인을 직접 구현한다.
3. LangSmith Trace를 활용해 프롬프트/체인/Agent 동작을 분석해본다.
4. FastAPI + LangServe로 RAG 체인을 API 형태로 배포하고 Insomnia로 호출해본다.

---

## 2. 사용 데이터

`data/docs/` 폴더 내 NIST PDF 3종:

- `nist_ai_risk_framework.pdf`
- `nist_cybersecurity_framework.pdf`
- `nist_zero_trust.pdf`

→ 모두 공공 정책·표준 문서로, RAG 실습에 적합한 도메인 텍스트를 제공.

---

## 3. 세부 커리큘럼 & 코드 매핑

### 3.1 LangChain 배경 이해 (0.5h)

- **내용**
  - LLM 시대의 애플리케이션 구조 변화
  - LLM 기반 앱 개발의 어려움: 프롬프트 관리, 체인 복잡성, 메모리/상태, 디버깅
  - LangChain Application Framework 개요
- **코드**
  - `src/01_langsmith_setup.py`
    - .env 로드
    - LangSmith Tracing V2 환경 변수 확인
    - 가장 간단한 LCEL 체인 실행 → Trace 생성 및 UI 확인

---

### 3.2 LangFamily 개요 (0.5h)

- **내용**
  - LangChain 주요 컴포넌트: PromptTemplate, LLM, Chains, Agents, Memory
  - LangSmith (Observability & Debugging)
  - LangServe (API 배포)
  - LangGraph (멀티에이전트/워크플로우) 개념 소개
  - LangChain Expression Language (LCEL) 기본 개념
- **코드**
  - `src/02_prompt_basics.py`
    - ChatPromptTemplate + ChatOpenAI + LCEL 체인
    - Trace 트리에서 Prompt와 LLM 호출 구조 관찰

---

### 3.3 활용 사례 소개 (0.5h)

- **내용**
  - 문서 요약, SQL 질의응답, 사내 정책봇, 다중 에이전트 예시
  - “PDF → RAG → 간단 Agent → API 배포” 전체 여정 설명
- **실습 목표**
  - NIST PDF 기반 RAG 챗봇 구현
  - LangSmith로 프롬프트/쿼리별 Trace 분석
  - FastAPI/LangServe로 배포 후 Insomnia에서 질의

---

### 3.4 기본 실습 (초급, 1.5h)

#### 3.4.1 Prompt & LLM / Chain (0.5h)

- **내용**
  - PromptTemplate / ChatOpenAI 기본 사용
  - LLMChain, SimpleSequentialChain 동작 이해
- **코드**
  - `src/03_chain_memory.py` (`run_sequential_chain`)
    - 요약 → 스타일 변경 두 단계 체인
    - Trace 상에서 두 번의 LLM 호출 구조 확인

#### 3.4.2 Memory & 대화형 챗봇 (1.0h)

- **내용**
  - ConversationBufferMemory 개념
  - 단순 Q&A에서 멀티턴 대화로 확장하는 방법
- **코드**
  - `src/03_chain_memory.py` (`run_conversation_chain`)
    - ConversationChain + Memory
    - Trace에서 대화 히스토리가 prompt로 어떻게 전달되는지 분석

---

### 3.5 응용 실습 (중급, 2h)

#### 3.5.1 텍스트 임베딩 & Chroma Vector Store 구축 (0.5h)

- **내용**
  - PDF 로딩 → 페이지 단위 Document 생성
  - Chunking 전략 (chunk_size, chunk_overlap)
  - OpenAIEmbeddings 소개
  - Chroma Vector Store 개념 (로컬 저장, collection_name)
- **코드**
  - `src/04_embeddings_chroma.py`
    - `load_all_pdfs()`: NIST PDF 3종 로드
    - `split_documents()`: RecursiveCharacterTextSplitter 사용
    - `build_chroma_index()`: Chroma.from_documents + persist

#### 3.5.2 Retriever + LLMChain 연결 / RAG 파이프라인 (1.5h)

- **내용**
  - Retriever 패턴: `.as_retriever(k=4)`
  - LCEL 기반 RAG 파이프라인 구성 (context + question)
  - 환각 최소화를 위한 prompt 설계
- **코드**
  - `src/rag_pipeline.py`
    - `get_retriever()`: Persisted Chroma 로드
    - `get_rag_chain()`: RunnableParallel + Prompt + ChatOpenAI
    - `rag_chain`: 전역 객체로 공개, 다양한 모듈에서 재사용
  - 실습:
    - Zero Trust / Cybersecurity / AI Risk 관련 질문 3~5개씩 실행
    - LangSmith에서 RAG 체인 Trace 구조 분석

---

### 3.6 심화 사례 & 마무리 (1h)

#### 3.6.1 Agent & ReAct 패턴 / Tool 연동 (0.4h)

- **내용**
  - Agent, Tool, ReAct 패턴 구조
  - LLM이 스스로 도구를 선택하고 사용하는 흐름
- **코드**
  - `src/06_agent_tools.py`
    - `@tool add(a, b)`: 간단 계산기
    - LangChain Hub `react` 프롬프트 템플릿 사용
    - create_react_agent + AgentExecutor
    - Trace에서 Agent → Tool → LLM 호출 경로 확인

#### 3.6.2 LangSmith 실험 추적 확인 (0.2h)

- **내용**
  - 프로젝트 필터(`LANGCHAIN_PROJECT`) 사용
  - RAG 쿼리별 품질 비교
  - 프롬프트 수정 전/후 run 비교
- **실습**
  - 참가자별 “가장 마음에 든 답변 / 마음에 들지 않은 답변” Trace 1개씩 공유
  - 개선 아이디어 토론

#### 3.6.3 LangServe로 API 배포 개념 & 데모 (0.4h)

- **내용**
  - LangServe의 역할: 체인을 HTTP API로 쉽게 노출
  - FastAPI와의 통합 구조, `/rag/invoke` / `/rag/playground` 엔드포인트
- **코드**
  - `api/serve_app.py`
    - FastAPI 앱 생성
    - `add_routes(app, rag_chain, path="/rag")`
    - `uvicorn`으로 실행
  - 추가 비교:
    - `src/07_api_fastapi.py`: LangServe 없이 순수 FastAPI 구현 예시
  - Insomnia 실습:
    - POST `http://localhost:8000/rag/invoke`
    - Body: `{ "question": "Zero Trust Architecture의 핵심 원칙은?" }`
    - 응답 및 Trace 동시 확인

---

## 4. 수업 전 사전 세팅 체크리스트

1. Python 3.10 또는 3.11 설치 및 PATH 설정
2. Git / VS Code(또는 선호 IDE) 설치
3. Repository 클론 및 가상환경 생성
4. `pip install -r requirements.txt`
5. `.env` 생성 후
   - `OPENAI_API_KEY`
   - `LANGCHAIN_API_KEY`
   - `LANGCHAIN_TRACING_V2=true`
   - `LANGCHAIN_PROJECT=gccare-rag-workshop`
   입력
6. `python src/01_langsmith_setup.py` 실행 → LangSmith Trace 생성 여부 확인

---

## 5. 기대 산출물

- PDF 기반 RAG Q&A 파이프라인 완성 코드
- Agent + Tool 연동 예제 코드
- FastAPI + LangServe 기반 RAG API 서버
- LangSmith 프로젝트 내 Trace / 실험 기록
- 사내 PoC로 확장 가능한 구조적 이해

---

본 계획서는 `gccare-langchain-rag-ai/docs/강의계획서.md` 로 저장하여  
GitHub에서 바로 열람할 수 있도록 제공합니다.
