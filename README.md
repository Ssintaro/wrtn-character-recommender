# 뤼튼 크랙 지능형 AI 캐릭터 추천 시스템

뤼튼 크랙(Wrtn Character) 플랫폼을 위한 **지능형 AI 캐릭터 추천 시스템**입니다. 사용자의 자연어 발화 맥락을 분석하여, 수많은 캐릭터 중 최적의 대화 상대를 매칭하는 RAG(Retrieval-Augmented Generation) 기반 추천 엔진입니다.

> 본 저장소는 컴퓨터 소프트웨어 공학 캡스톤 디자인 프로젝트의 산출물이며, 현재 **1차 프로토타입(v1)** 단계입니다. 자세한 구현 결과와 한계점 진단은 [`docs/prototype_v1_report.md`](docs/prototype_v1_report.md)를 참고하세요.

## 프로젝트 개요

기존 태그 기반 검색의 한계(선택의 역설, 시맨틱 갭, 인기 편향)를 극복하기 위해, LLM과 벡터 검색을 결합한 하이브리드 추천 파이프라인을 구현합니다. 사용자는 수동으로 카테고리를 고르는 대신, 자연스러운 문장으로 원하는 바를 말하면 됩니다.

**예시 질의:** *"요즘 마감 때문에 밤새서 피곤해. 이세계 전생해서 모험하는 시뮬레이션 같은 거 없나? 연애 위주 로맨스는 싫어."*

## 추천 파이프라인 (5단계)

| 단계 | 명칭 | 기능 |
|------|------|------|
| ① | Static Profile 매칭 | 고정된 사용자 성향으로 1차 하드 필터링 |
| ② | Query Intent Parsing | 발화에서 장르·긍정/부정 카테고리 의도를 구조화 추출 |
| ③ | Negative Screening | 기피 카테고리 포함 후보를 애플리케이션 레이어에서 배제 |
| ④ | 3-Axis Reranking | 정규화 유사도·최신성·카테고리 매칭의 가중 합산 재정렬 |
| ⑤ | Generation | 최종 1위 캐릭터에 빙의한 1인칭 추천 멘트 생성 |

## 기술 스택

- **언어:** Python 3.13
- **생성 LLM:** Claude Haiku 4.5 (의도 파싱 · 추천 멘트 생성)
- **임베딩:** OpenAI `text-embedding-3-small`
- **벡터 저장소:** ChromaDB (로컬 파일 기반 영구 저장)
- **설계 원칙:** LangChain 등 프레임워크 없이 raw SDK만 사용하여 파이프라인을 투명하게 제어

## 프로젝트 구조

```
wrtn-character-recommender/
├── data/
│   └── mock/
│       └── characters.json   # 뤼튼 크랙 미러링 합성 데이터셋 (50개)
├── src/
│   ├── embedding/            # 임베딩 생성 (OpenAIEmbedder)
│   ├── store/                # 벡터 저장소 관리 (CharacterVectorStore)
│   └── recommender/          # 추천 엔진 (CharacterRecommenderEngine)
├── scripts/
│   ├── build_index.py        # 데이터 적재 및 검색 테스트
│   └── recommend.py          # 추천 파이프라인 실행
├── docs/                     # 프로젝트 문서
├── .env.example              # API 키 템플릿
└── requirements.txt
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
```

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

- Anthropic 키 발급: https://console.anthropic.com
- OpenAI 키 발급: https://platform.openai.com

> `.env` 파일은 `.gitignore`에 등록되어 있어 저장소에 올라가지 않습니다. API 키는 절대 코드나 커밋에 포함하지 마세요.

### 4. 실행

```bash
# ① 데이터셋을 ChromaDB에 적재 (최초 1회 또는 데이터 변경 시)
py -m scripts.build_index

# ② 추천 파이프라인 실행
py -m scripts.recommend
```

## 데이터셋 안내

`data/mock/characters.json`의 50개 캐릭터 데이터는 실제 뤼튼 크랙 서비스가 아닌 **합성 데이터(Synthetic Data)** 입니다. 실서비스 화면 분석을 토대로 데이터의 스키마·메트릭 분포·텍스트 뉘앙스를 미러링하여, 추천 알고리즘의 변별력 검증을 목적으로 구축되었습니다.

## 개발 현황

- **v1 (현재):** 매칭 엔진 아키텍처 골격 수립, 50개 미러링 데이터셋 기반 검증 완료
- **v2 (예정):** 매칭 품질 개선(임베딩 텍스트 보강, 다각도 카테고리 매칭), 자동화 데이터 파이프라인 구축, 정량적 평가 체계(RAGAS) 도입

자세한 내용은 [`docs/prototype_v1_report.md`](docs/prototype_v1_report.md)를 참고하세요.

## 라이선스

본 프로젝트는 학술 목적의 캡스톤 디자인 산출물입니다.
