# 뤼튼 크랙 지능형 AI 캐릭터 추천 시스템 v2

뤼튼 크랙(Wrtn Character) 플랫폼을 위한 **대화 기반 하이브리드 캐릭터 추천 시스템**입니다. 사용자의 자연어 발화 맥락을 멀티턴으로 누적·분석하여, 의미 매칭·태그 커버리지·품질·최신성의 4축 가중 합산으로 최적 캐릭터를 선정하고, 선정된 캐릭터와 실시간 RAG 대화를 이어갈 수 있는 End-to-End 시스템입니다.

> 본 저장소는 컴퓨터 소프트웨어 공학 캡스톤 디자인 프로젝트의 산출물이며, 현재 **2차 프로토타입(v2.0.0)** 단계입니다. 1차 프로토타입의 결함 진단과 그에 대한 응답은 [`docs/prototype_v1_report.md`](docs/prototype_v1_report.md)와 [`docs/prototype_v2_report.md`](docs/prototype_v2_report.md)를 참고하세요.

## 프로젝트 개요

기존 태그 기반 검색의 한계(선택의 역설, 시맨틱 갭, 인기 편향)를 극복하기 위해, LLM과 벡터 검색을 결합한 하이브리드 추천 파이프라인을 구현합니다. 사용자는 수동으로 카테고리를 고르는 대신, 자연스러운 문장으로 원하는 바를 말하면 됩니다.

2차 프로토타입은 1차에서 식별된 매칭 결함을 4축 가중 합산 랭킹과 태그 커버리지 매칭으로 극복하였으며, 동적 크롤링·증분 적재·실시간 RAG 대화로 이어지는 자동화 파이프라인과, OpenAI GPT 판정자를 활용한 LLM-as-a-Judge 평가 하니스를 갖추었습니다. v4 평가 결과 종합 평균 **3.48 / 5.00**을 달성했습니다.

**예시 시나리오 (멀티턴):**
- 1턴: *"요즘 너무 지쳐서 빡센 생존 액션 시뮬레이션이 땡겨. 던전 같은 거 들어가서 살아남는 거."*
- 2턴: *"아 그리고 나 말랑말랑한 로맨스풍 요소 섞인 건 진짜 싫어해. 빼줘."*
- → 시스템이 1턴의 긍정 취향을 유지한 채 2턴의 기피 조건을 누적 반영하여 추천을 갱신.

## 시스템 아키텍처

전체 파이프라인은 다음과 같이 결합된 단방향 데이터 흐름을 따릅니다.

```
[크롤러]                                       (v3, src/crawler)
    ↓ raw 데이터
[정규화 파이프라인]                              (v3, src/store)
    ↓ 스키마 v2 규격 데이터
[벡터 저장소 ChromaDB]                          (v2/v3, src/store)
    ↑ ↓ 임베딩 · 메타데이터 (멱등 upsert)
[추천 엔진]                                    (v2, src/recommender/engine)
    ← 누적 유저 프로필 (v2, src/recommender/profile_manager)
    ↓ 1위 캐릭터 + 추천 멘트
[RAG 대화 루프]                                (v3, chat_with_character)
    ↓ 추천 결과
[평가 하니스]                                  (v4, src/evaluation/harness)
    ← OpenAI GPT 판정자 (교차 검증)
```

## 추천 파이프라인 (6단계)

| 단계 | 명칭 | 기능 |
|------|------|------|
| ① | Multi-turn 프로파일링 | 유저 발화 누적 → 긍정/기피 태그, 선호 장르, 정서 상태 갱신 |
| ② | Static Profile 매칭 | 고정 성향으로 1차 하드 필터링 (`unspecified` 통합 허용) |
| ③ | 절대 유사도 가드 | raw 유사도 임계값 미달 후보를 노이즈로 간주, 사전 차단 |
| ④ | Negative Screening | 누적 기피 태그가 포함된 후보를 애플리케이션 레이어에서 배제 |
| ⑤ | 4-Axis Reranking | 정규화 유사도 · 태그 커버리지 · 품질 · 최신성의 가중 합산 |
| ⑥ | RAG 대화 | 최종 1위 캐릭터에 빙의한 1인칭 멀티턴 연속 대화 |

**4축 가중 합산 공식:**
```
final_score = 0.35 × normalized_similarity     (유사도)
            + 0.30 × tag_coverage              (태그 커버리지)
            + 0.20 × normalized_quality        (품질 — 베이지안 보정 좋아요 비율)
            + 0.15 × recency_score             (최신성 — 30일 선형 감쇠)
```

## 기술 스택

- **언어:** Python 3.13
- **생성 LLM:** Claude Haiku 4.5 (`claude-haiku-4-5-20251001`) — 의도 파싱 · 추천 멘트 · RAG 대화
- **임베딩:** OpenAI `text-embedding-3-small`
- **벡터 저장소:** ChromaDB (로컬 파일 기반 영구 저장)
- **동적 크롤링:** Playwright (Chromium headless)
- **자동 평가:** OpenAI GPT (`gpt-5.4-mini`) 기반 LLM-as-a-Judge (Structured Outputs)
- **설계 원칙:** LangChain·RAGAS 등 프레임워크 없이 raw SDK만 사용하여 파이프라인을 투명하게 제어

## 프로젝트 구조

```
wrtn-character-recommender/
├── data/
│   ├── mock/
│   │   └── characters.json       # 50개 합성 데이터셋 (뤼튼 크랙 미러링)
│   └── mock_site/
│       └── characters.html       # 가상 크롤링 타깃 (실서비스 DOM 모사)
├── src/
│   ├── crawler/                  # Playwright 기반 동적 크롤러
│   │   └── collector.py
│   ├── embedding/                # OpenAI 임베딩 래퍼
│   │   └── embedder.py
│   ├── store/                    # ChromaDB 적재·검색 + 크롤링 데이터 정규화
│   │   └── vector_store.py
│   ├── recommender/              # 4축 랭킹 엔진 + 다턴 프로파일러
│   │   ├── engine.py
│   │   └── profile_manager.py
│   └── evaluation/               # LLM-as-a-Judge 평가 하니스
│       └── harness.py
├── scripts/
│   ├── build_index.py            # 50개 합성 데이터셋 초기 적재
│   ├── main_pipeline_v3.py       # 크롤링→적재→추천→RAG 대화 통합 실행
│   └── run_evaluation.py         # 7개 시나리오 자동 평가
├── docs/
│   ├── prototype_v1_report.md    # 1차 프로토타입 보고서
│   └── prototype_v2_report.md    # 2차 프로토타입 보고서
├── .env.example                  # API 키 템플릿
├── .gitignore
├── requirements.txt
└── README.md
```

## 설치 및 실행

### 1. 저장소 클론 및 가상환경 준비

```bash
git clone https://github.com/Ssintaro/wrtn-character-recommender.git
cd wrtn-character-recommender
py -3.13 -m venv .venv
.venv\Scripts\activate
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
playwright install chromium
```

> `playwright install chromium`은 동적 크롤링용 브라우저 엔진(수십 MB)을 받는 1회성 명령입니다.

### 3. API 키 설정

`.env.example`을 복사해 `.env` 파일을 만들고, 본인의 API 키 두 개를 채웁니다.

```bash
copy .env.example .env
```

`.env` 파일 내용:
```
ANTHROPIC_API_KEY=본인의_Anthropic_키
OPENAI_API_KEY=본인의_OpenAI_키
```

- Anthropic 키 발급: <https://console.anthropic.com>
- OpenAI 키 발급: <https://platform.openai.com>

> `.env` 파일은 `.gitignore`에 등록되어 있어 저장소에 올라가지 않습니다. API 키는 절대 코드나 커밋에 포함하지 마세요.

### 4. 실행 시나리오

```bash
# 시나리오 A — 50개 합성 데이터셋만으로 시작 (최초 1회)
py -m scripts.build_index

# 시나리오 B — 전체 v3 파이프라인 (크롤링부터 RAG 대화까지)
py -m scripts.main_pipeline_v3

# 시나리오 C — LLM-as-a-Judge 자동 평가 (7개 시나리오 채점)
py -m scripts.run_evaluation
```

## 데이터셋 안내

`data/mock/characters.json`의 50개 캐릭터 데이터는 실제 뤼튼 크랙 서비스가 아닌 **합성 데이터(Synthetic Data)** 입니다. 실서비스 화면 분석을 토대로 데이터의 스키마·메트릭 분포·텍스트 뉘앙스를 정밀하게 미러링하여, 추천 알고리즘의 변별력 검증을 목적으로 구축되었습니다.

`data/mock_site/characters.html`은 동적 크롤링 시연용 **가상 미러링 타깃**입니다. 실서비스 DOM 구조(스토리 정보 레이아웃)를 모사하여, 크롤러의 아키텍처 검증이 외부 의존성 없이 재현 가능하도록 구성되었습니다.

## 평가 결과 (v4 하니스)

OpenAI GPT 판정자가 채점한 7개 정밀 실험 시나리오의 평균 점수:

| 지표 | 평균 (1~5점) |
|------|--------------|
| 적합성 (Relevance) | 3.43 |
| 설명력 (Explainability) | 3.00 |
| 사실 충실성 (Faithfulness) | 4.00 |
| **종합** | **3.48** |

지표별 강·약점에 대한 구조적 진단은 [2차 프로토타입 보고서 5장](docs/prototype_v2_report.md)을 참고하세요.

## 개발 현황

- **v1 (1차 프로토타입)**: 매칭 엔진 골격 + 50개 합성 데이터셋
- **v2.0.0 (2차 프로토타입, 현재)**: 4축 랭킹 + Multi-turn 프로파일링 + 동적 크롤링 파이프라인 + RAG 대화 + LLM-as-a-Judge 평가
- **v3 (예정)**: 긍정 의도 매칭 강화, 한국어 특화 임베딩 검토, 자동 스케줄링 기반 무인 갱신

## 라이선스

본 프로젝트는 학술 목적의 캡스톤 디자인 산출물입니다.
