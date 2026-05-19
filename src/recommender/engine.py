"""src/recommender/engine.py

역할: 5단계 하이브리드 추천 파이프라인을 관장하는 CharacterRecommenderEngine.

파이프라인:
  ① Static Profile 매칭   : 고정 성향으로 하드 필터
  ② Query Intent Parsing  : Claude로 유저 의도(장르/긍정/부정 카테고리) 추출
  ③ 네거티브 스크리닝     : 기피 카테고리 묻은 후보를 파이썬 단에서 배제
  ④ 3축 랭킹             : 유사도 + 최신성 + 카테고리매칭 가중 합산
  ⑤ 추천 멘트 생성        : 1위 캐릭터에 빙의해 Claude가 대화 생성
"""
import json
from datetime import date, datetime
from pathlib import Path

from anthropic import Anthropic

from src.store.vector_store import CharacterVectorStore

# 사용할 LLM (의도 파싱 · 멘트 생성 공통).
LLM_MODEL = "claude-haiku-4-5-20251001"

# ★ category 필드가 가질 수 있는 값의 '공식 화이트리스트'.
#   인텐트 파서가 이 목록 밖 단어를 뱉으면 카테고리 매칭이 무력화되므로,
#   파서 프롬프트에 이 목록을 박고, 코드에서도 이 목록으로 한 번 더 거른다.
VALID_CATEGORIES = [
    "romance", "bl", "gl", "sf_fantasy", "romance_fantasy",
    "martial_arts", "daily_modern", "period", "adult", "etc",
]


class CharacterRecommenderEngine:
    """뤼튼 크랙 지능형 하이브리드 추천 엔진."""

    def __init__(
        self,
        weight_similarity: float = 0.6,
        weight_recency: float = 0.15,
        weight_category: float = 0.25,
        recency_window_days: int = 30,
        reference_date: date | None = None,
    ):
        """
        weight_* : 3축 최종 점수의 가중치. 합이 1.0 이 되도록 설계.
            발표 시 이 값들을 바꿔 '어느 축을 중시할지'를 시연할 수 있다.
        recency_window_days : 신작으로 간주해 최신성 보너스를 주는 기준 창.
        reference_date : 최신성 계산 기준일. 명세대로 2026-05-19 로 고정 주입.
            None 이면 실행 당일을 쓰지만, 발표 재현성을 위해 고정을 권장.
        """
        self.store = CharacterVectorStore()
        self.llm = Anthropic()  # 환경변수 ANTHROPIC_API_KEY 자동 인식

        self.w_sim = weight_similarity
        self.w_rec = weight_recency
        self.w_cat = weight_category
        self.recency_window_days = recency_window_days
        self.reference_date = reference_date or date.today()

        # 1위 캐릭터의 '전체 프로필'은 멘트 생성에 필요하므로,
        # 원본 JSON을 id 기준 조회 사전으로 미리 로드해 둔다.
        self._profiles = self._load_profile_lookup()

    # ------------------------------------------------------------------
    # 준비: 원본 JSON을 id 기준 조회 사전으로 로드
    # ------------------------------------------------------------------
    @staticmethod
    def _load_profile_lookup() -> dict[str, dict]:
        """data/mock/characters.json 을 {character_id: 캐릭터dict} 로 변환."""
        # __file__ = .../src/recommender/engine.py → parents[2] = 프로젝트 루트.
        root = Path(__file__).resolve().parents[2]
        data_path = root / "data" / "mock" / "characters.json"
        with open(data_path, encoding="utf-8") as f:
            characters = json.load(f)
        return {str(c["character_id"]): c for c in characters}

    # ==================================================================
    # ② Query Intent Parsing — Claude로 유저 의도 추출
    # ==================================================================
    def _parse_intent(self, user_query: str) -> dict:
        """
        유저의 자연어 질문을 Claude에 던져, 구조화된 의도를 JSON으로 받는다.

        반환 형식:
          {"genre": "simulation"|"roleplay_1on1"|null,
           "categories_positive": [...],   # VALID_CATEGORIES 부분집합
           "categories_negative": [...]}   # VALID_CATEGORIES 부분집합

        ★ 핵심 설계: system 프롬프트에 허용 카테고리 10개를 명시하고,
          "목록 밖 단어 금지"를 못박는다. LLM이 'action' 같은 자유 단어를
          뱉으면 이후 카테고리 매칭이 조용히 무력화되기 때문이다.
        """
        category_list_str = ", ".join(VALID_CATEGORIES)

        system_prompt = (
            "너는 유저의 자연어 문장에서 콘텐츠 추천 의도를 추출하는 "
            "파서다. 반드시 아래 규칙을 지켜라.\n"
            f"1. genre 는 'roleplay_1on1' 또는 'simulation' 중 하나, "
            "또는 알 수 없으면 null.\n"
            f"2. categories_positive / categories_negative 의 원소는 "
            f"반드시 다음 목록 안의 값만 사용한다: [{category_list_str}].\n"
            "   이 목록에 없는 단어(예: action, healing 등)는 절대 쓰지 마라.\n"
            "3. 유저가 명시적으로 싫다고 한 소재만 negative 에 넣는다.\n"
            "4. 출력은 오직 JSON 객체 하나. 설명·코드블록·여는말 금지."
        )

        user_prompt = (
            f'다음 문장을 분석해 JSON으로만 답하라.\n\n문장: "{user_query}"\n\n'
            '형식: {"genre": ..., "categories_positive": [...], '
            '"categories_negative": [...]}'
        )

        message = self.llm.messages.create(
            model=LLM_MODEL,
            max_tokens=300,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw_text = message.content[0].text.strip()

        # LLM이 실수로 ```json 펜스를 붙일 수 있으니 방어적으로 제거.
        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`")
            # "json\n{...}" 형태가 되면 첫 줄(json) 제거.
            if raw_text.lower().startswith("json"):
                raw_text = raw_text[4:]

        # JSON 파싱. 실패 시 '의도 없음'으로 안전하게 폴백.
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            print("⚠️  인텐트 파싱 실패 → 의도 없이 진행합니다.")
            parsed = {}

        # --- 결과 정제 (LLM 출력을 그대로 믿지 않는다) ---
        genre = parsed.get("genre")
        if genre not in ("roleplay_1on1", "simulation"):
            genre = None  # 허용값 외 → None

        # 긍정/부정 카테고리를 VALID_CATEGORIES 화이트리스트로 교차 검증.
        def _clean(cats) -> list[str]:
            if not isinstance(cats, list):
                return []
            return [c for c in cats if c in VALID_CATEGORIES]

        return {
            "genre": genre,
            "categories_positive": _clean(parsed.get("categories_positive")),
            "categories_negative": _clean(parsed.get("categories_negative")),
        }

    # ==================================================================
    # ③ 네거티브 스크리닝 — 기피 카테고리 묻은 후보 배제
    # ==================================================================
    @staticmethod
    def _screen_negative(
        candidates: list[dict], categories_negative: list[str]
    ) -> list[dict]:
        """
        후보의 category 문자열을 검사해, 기피 카테고리가 1개라도 묻어 있으면
        탈락시킨다. (ChromaDB는 문자열 부분일치 필터가 없어 파이썬 단에서 처리.)

        category 는 "romance,daily_modern" 같은 쉼표 결합 문자열이므로,
        쉼표로 쪼개 집합(set)으로 만든 뒤 교집합 여부를 본다.
        """
        if not categories_negative:
            return candidates  # 기피 조건이 없으면 그대로 통과.

        neg_set = set(categories_negative)
        survivors = []
        for cand in candidates:
            # "romance,daily_modern" → {"romance", "daily_modern"}
            cat_set = {
                c.strip()
                for c in cand["metadata"].get("category", "").split(",")
                if c.strip()
            }
            # 교집합이 비어 있어야(기피 소재가 안 묻어야) 생존.
            if cat_set & neg_set:
                continue  # 기피 소재 발견 → 탈락
            survivors.append(cand)
        return survivors

    # ==================================================================
    # ④ 3축 랭킹 — 유사도 + 최신성 + 카테고리매칭
    # ==================================================================
    def _recency_score(self, updated_at: str) -> float:
        """
        updated_at 을 기준일과 비교해 0~1 의 최신성 점수로 변환 (선형 감쇠).
          days_old = 기준일 - updated_at
          score    = max(0, 1 - days_old / window)
        오늘 갱신=1.0, 창의 절반 경과=0.5, 창(30일) 초과=0.0.
        """
        updated = datetime.strptime(updated_at, "%Y-%m-%d").date()
        days_old = (self.reference_date - updated).days
        days_old = max(0, days_old)  # 미래 날짜 방어
        score = 1.0 - (days_old / self.recency_window_days)
        return max(0.0, min(1.0, score))

    @staticmethod
    def _category_match_score(
        character_category: str, categories_positive: list[str]
    ) -> float:
        """
        캐릭터 category 와 유저 긍정 카테고리의 '교집합 비율'을 0~1로 계산.
          score = (겹치는 개수) / (유저가 원한 긍정 카테고리 개수)
        유저가 긍정 카테고리를 하나도 안 줬으면 비교 불가 → 중립값 0.0.
          (0.0 으로 두면 카테고리 축이 점수에 영향을 안 주게 되어 안전)
        """
        if not categories_positive:
            return 0.0
        char_set = {
            c.strip() for c in character_category.split(",") if c.strip()
        }
        pos_set = set(categories_positive)
        overlap = char_set & pos_set
        return len(overlap) / len(pos_set)

    def _rerank(
        self, candidates: list[dict], categories_positive: list[str]
    ) -> list[dict]:
        """
        후보군을 3축 가중 합산으로 재정렬한다.

        최종점수 = w_sim·정규화유사도 + w_rec·최신성 + w_cat·카테고리매칭

        ★ 정규화유사도: OpenAI 임베딩의 raw 유사도는 후보 간 격차가
          0.03 수준으로 압축돼 있다. 후보군 내부에서 Min-Max Scaling 하여
          [0,1]로 강제로 펴야 다른 두 축과 같은 무대에 설 수 있다.
        """
        # --- 1단계: 각 후보의 raw 점수 3종을 먼저 모두 계산 ---
        # 정규화는 후보군 전체의 Min/Max 가 필요하므로 한 번에 훑는다.
        for cand in candidates:
            cand["_raw_sim"] = max(0.0, 1.0 - cand["distance"])
            cand["_recency"] = self._recency_score(
                cand["metadata"]["updated_at"]
            )
            cand["_catmatch"] = self._category_match_score(
                cand["metadata"].get("category", ""), categories_positive
            )

        # --- 2단계: 후보군 내 raw 유사도의 Min/Max 추출 ---
        sims = [c["_raw_sim"] for c in candidates]
        sim_min, sim_max = min(sims), max(sims)
        sim_range = sim_max - sim_min  # 정규화 분모

        # --- 3단계: 정규화 + 3축 가중 합산 ---
        scored = []
        for cand in candidates:
            raw_sim = cand["_raw_sim"]

            # ★ 분모 0 방어: 후보들의 유사도가 전부 동일하면 우열 불가
            #   → 모두 중립값 0.5 부여 (특정 극단으로 쏠리지 않게).
            if sim_range == 0:
                norm_sim = 0.5
            else:
                norm_sim = (raw_sim - sim_min) / sim_range

            recency = cand["_recency"]
            catmatch = cand["_catmatch"]

            final = (
                self.w_sim * norm_sim
                + self.w_rec * recency
                + self.w_cat * catmatch
            )

            cand_scored = dict(cand)
            cand_scored["raw_similarity"] = round(raw_sim, 4)
            cand_scored["norm_similarity"] = round(norm_sim, 4)
            cand_scored["recency_score"] = round(recency, 4)
            cand_scored["category_score"] = round(catmatch, 4)
            cand_scored["final_score"] = round(final, 4)
            scored.append(cand_scored)

        scored.sort(key=lambda c: c["final_score"], reverse=True)
        return scored

    # ==================================================================
    # 발표용: 재정렬 점수표를 터미널에 줄 맞춰 출력
    # ==================================================================
    @staticmethod
    def print_ranking_table(ranking: list[dict]) -> None:
        """[raw유사도, 정규화, 최신성, 카테고리, 최종점수]를 표로 출력."""
        print("\n=== 📊 재정렬 점수표 (3축 하이브리드 랭킹) ===")
        header = (
            f"{'캐릭터명':<22}{'raw유사':>9}{'정규화':>9}"
            f"{'최신성':>9}{'카테고리':>10}{'최종점수':>11}"
        )
        print(header)
        print("-" * len(header))
        for r in ranking:
            name = r["metadata"]["name"]
            # 한글 폭 보정: 길면 잘라서 표가 안 깨지게.
            display_name = name if len(name) <= 20 else name[:19] + "…"
            print(
                f"{display_name:<22}"
                f"{r['raw_similarity']:>9.4f}"
                f"{r['norm_similarity']:>9.4f}"
                f"{r['recency_score']:>9.4f}"
                f"{r['category_score']:>10.4f}"
                f"{r['final_score']:>11.4f}"
            )

    # ==================================================================
    # ⑤ 추천 멘트 생성 — 1위 캐릭터에 빙의
    # ==================================================================
    def _generate_message(self, profile: dict, user_query: str) -> str:
        """선정된 캐릭터/세계관에 빙의해 Claude가 추천 대화를 생성한다."""
        system_prompt = (
            f"너는 '{profile['name']}'라는 AI 콘텐츠다. "
            f"장르: {profile.get('genre', '')}, "
            f"분위기 태그: {profile.get('tags_raw', '')}. "
            "이 작품의 화자가 되어, 작품 특유의 말투와 분위기로 1인칭으로 말하라."
        )
        user_prompt = f"""[너의 프로필]
- 이름/세계관: {profile['name']}
- 카테고리: {profile.get('category', '')}
- 감성 키워드: {profile.get('tags_normalized', '')}
- 제작자: {profile.get('creator_name', '')}

[지금 너를 만난 사용자의 상황]
{user_query}

[지시]
위 사용자에게, 왜 지금 너(이 작품/캐릭터)가 그에게 어울리는지를
2~3문장의 매력적인 대화체로, 작품의 분위기를 살려 직접 말을 건네듯 표현하라.
설명문이 아니라 작품 속 화자가 사용자를 초대하는 첫 마디처럼."""

        message = self.llm.messages.create(
            model=LLM_MODEL,
            max_tokens=400,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return message.content[0].text

    # ==================================================================
    # 공개 메서드: 5단계 파이프라인 전체 실행
    # ==================================================================
    def recommend(self, user_query: str, target_orientation: str) -> dict:
        """
        유저 질문 + 고정 성향 → 추천 결과까지 5단계를 한 번에 실행한다.

        반환 dict:
          intent       : 파싱된 의도
          ranking      : 재정렬된 최종 후보(점수 내역 포함)
          winner       : 1위 캐릭터 전체 프로필
          message      : Claude 생성 추천 멘트
          fallback_used: 네거티브 배제 결과가 0건이라 폴백했는지 여부
        """
        # ② 인텐트 파싱.
        intent = self._parse_intent(user_query)
        print(f"🧠 파싱된 의도: {intent}")

        # ① Static Profile 매칭 + 동적 장르 필터로 후보 15명 추출.
        candidates = self.store.search_characters(
            query_text=user_query,
            target_orientation=target_orientation,
            genre=intent["genre"],
            top_k=15,
        )
        if not candidates:
            raise ValueError(
                "하드 필터 결과 후보가 0건입니다. "
                "성향/장르 조건이 너무 좁거나 DB가 비어있습니다."
            )

        # ③ 네거티브 스크리닝.
        screened = self._screen_negative(
            candidates, intent["categories_negative"]
        )

        # ★ 엣지 케이스 방어: 배제 후 후보가 0건이면,
        #   배제 전 후보군으로 폴백한다(추천이 아예 끊기는 것 방지).
        fallback_used = False
        if not screened:
            print("⚠️  기피 조건 배제 후 후보가 0건 → 배제 전 후보로 폴백.")
            screened = candidates
            fallback_used = True

        # ④ 3축 랭킹.
        ranking = self._rerank(screened, intent["categories_positive"])
        winner_id = ranking[0]["id"]
        winner_profile = self._profiles[winner_id]

        # ⑤ 추천 멘트 생성.
        message = self._generate_message(winner_profile, user_query)

        return {
            "intent": intent,
            "ranking": ranking,
            "winner": winner_profile,
            "message": message,
            "fallback_used": fallback_used,
        }