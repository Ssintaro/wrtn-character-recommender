"""src/recommender/engine.py  (v3)

역할: v3 대화 기반 하이브리드 추천 + 실시간 RAG 대화 엔진.

v2 대비 추가:
  - chat_with_character: 랭킹 1위 캐릭터에 빙의해 유저와 연속 멀티턴
    대화를 주고받는 RAG 챗봇 루프.

핵심 구조:
  - 단발성 인텐트 파싱 제거 → UserProfileManager 가 누적한 프로필 입력.
  - 4축 랭킹: 0.35·정규화유사도 + 0.30·태그매칭
            + 0.20·정규화품질  + 0.15·최신성
  - tags_normalized 기반 커버리지 매칭으로 변별력 확보.
  - 메트릭(좋아요/대화수) 기반 품질 축 + 베이지안 소표본 보정.
  - creator_name / creator_followers 는 인기 편향 억제를 위해
    추천 로직에서 일절 사용하지 않는다.
"""
import json
from datetime import date, datetime
from pathlib import Path

from anthropic import Anthropic

from src.store.vector_store import CharacterVectorStore

# 사용할 LLM (멘트 생성 · RAG 대화 공통).
LLM_MODEL = "claude-haiku-4-5-20251001"


class CharacterRecommenderEngine:
    """v3 대화 기반 하이브리드 추천 + RAG 대화 엔진."""

    def __init__(
        self,
        weight_similarity: float = 0.35,
        weight_tagmatch: float = 0.30,
        weight_quality: float = 0.20,
        weight_recency: float = 0.15,
        recency_window_days: int = 30,
        reference_date: date | None = None,
        threshold_guard: float = 0.25,
        quality_confidence_k: int = 50000,
    ):
        """
        4축 가중치 (합 1.0):
          similarity 0.35 + tagmatch 0.30 → 의미 매칭(0.65, 주축)
          quality 0.20  → 메트릭 기반 품질 (인기 편향 보정)
          recency 0.15  → 신작 우대 (롱테일)

        recency_window_days : 신작으로 간주해 최신성 보너스를 주는 창(일).
        reference_date : 최신성 계산 기준일. None 이면 실행 당일.
            발표 재현성을 위해 특정 날짜 고정 주입을 권장.
        threshold_guard : 벡터 검색 시 적용할 절대 유사도 하한선.
        quality_confidence_k : 품질 점수의 베이지안 보정 상수.
            대화수가 이 값에 한참 못 미치는 소형 작품일수록, 좋아요 비율을
            전체 평균 쪽으로 더 강하게 끌어당겨 소표본 편의를 억제한다.
        """
        self.store = CharacterVectorStore()
        self.llm = Anthropic()  # 환경변수 ANTHROPIC_API_KEY 자동 인식

        self.w_sim = weight_similarity
        self.w_tag = weight_tagmatch
        self.w_qual = weight_quality
        self.w_rec = weight_recency
        self.recency_window_days = recency_window_days
        self.reference_date = reference_date or date.today()
        self.threshold_guard = threshold_guard
        self.quality_confidence_k = quality_confidence_k

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
    # 네거티브 스크리닝 — 누적 기피 태그로 후보 배제
    # ==================================================================
    @staticmethod
    def _screen_negative(
        candidates: list[dict], accumulated_negatives: list[str]
    ) -> list[dict]:
        """
        후보의 tags_normalized 를 검사해, 누적 기피 태그가 1개라도
        묻어 있으면 탈락시킨다.

        v2 이상: '대화 전체에 걸쳐 누적된' accumulated_negatives 로 거른다.
        2턴에서 "로맨스 빼줘"가 누적되면, 이후 모든 추천에 영구 적용된다.
        """
        if not accumulated_negatives:
            return candidates  # 기피 조건이 없으면 그대로 통과.

        neg_set = set(accumulated_negatives)
        survivors = []
        for cand in candidates:
            tag_set = {
                t.strip()
                for t in cand["metadata"].get("tags_normalized", "").split(",")
                if t.strip()
            }
            # 교집합이 비어 있어야(기피 태그가 안 묻어야) 생존.
            if tag_set & neg_set:
                continue
            survivors.append(cand)
        return survivors

    # ==================================================================
    # 점수 계산 ① 최신성 점수
    # ==================================================================
    def _recency_score(self, updated_at: str) -> float:
        """
        updated_at("YYYY-MM-DD")을 기준일과 비교해 0~1 최신성 점수로 변환.

        수식 (선형 감쇠):
            days_old = 기준일 - updated_at
            score    = max(0, 1 - days_old / window)
        오늘 갱신=1.0, 창의 절반 경과=0.5, 창(30일) 초과=0.0.
        """
        updated = datetime.strptime(updated_at, "%Y-%m-%d").date()
        days_old = (self.reference_date - updated).days
        days_old = max(0, days_old)  # 미래 날짜(데이터 오류) 방어
        score = 1.0 - (days_old / self.recency_window_days)
        return max(0.0, min(1.0, score))

    # ==================================================================
    # 점수 계산 ② 태그 매칭 점수 (커버리지 방식)
    # ==================================================================
    @staticmethod
    def _tag_match_score(
        character_tags: str, accumulated_positives: list[str]
    ) -> float:
        """
        캐릭터 tags_normalized 와 유저 누적 긍정 태그의 '커버리지 비율'.

          score = (캐릭터가 보유한, 유저 선호 태그의 수) / (유저 선호 태그 총수)

        ★ 자카드(교집합/합집합) 대신 커버리지(교집합/유저선호수)를 쓰는 이유:
          캐릭터 태그(약 4개)와 유저 누적 선호(대화가 쌓이면 6~8개)는
          크기가 비대칭이다. 자카드는 합집합을 분모로 써서, 캐릭터가
          유저 선호를 완벽히 만족시켜도 점수가 낮게 깔린다.
          우리가 원하는 것은 '유사도'가 아니라 '유저 취향을 얼마나
          커버하는가'이므로, 분모를 유저 선호 수로 고정한 커버리지가 맞다.

        유저 긍정 태그가 없으면 비교 불가 → 0.0 (이 축이 점수에 무영향).
        """
        if not accumulated_positives:
            return 0.0
        char_set = {
            t.strip() for t in character_tags.split(",") if t.strip()
        }
        pos_set = set(accumulated_positives)
        overlap = char_set & pos_set
        return len(overlap) / len(pos_set)

    # ==================================================================
    # 점수 계산 ③ 품질 점수 (메트릭 기반, 베이지안 보정)
    # ==================================================================
    @staticmethod
    def _compute_global_mean_ratio(candidates: list[dict]) -> float:
        """
        후보군 전체의 평균 좋아요 비율(prior m)을 계산한다.
        베이지안 보정에서 '소형 작품을 끌어당길 기준점'으로 쓰인다.
        후보가 비거나 대화수가 모두 0이면 중립값 0.0 을 반환한다.
        """
        ratios = []
        for c in candidates:
            conv = int(c["metadata"].get("conversation_count", 0))
            like = int(c["metadata"].get("like_count", 0))
            if conv > 0:
                ratios.append(like / conv)
        if not ratios:
            return 0.0
        return sum(ratios) / len(ratios)

    def _quality_score(
        self, candidate: dict, global_mean_ratio: float
    ) -> float:
        """
        메트릭 기반 품질 점수(베이지안 보정된 raw 비율)를 산출한다.

        설계 근거 (보고서 기재용):
          '대화 생성 수'를 직접 쓰면 누적 규모가 큰 네임드가 무조건
          이겨 인기 편향이 심화된다(계획서가 지적한 핵심 문제).
          따라서 규모와 무관한 '만족도 비율'을 쓴다:
              raw_ratio = like_count / conversation_count

          ★ 소표본 보정 (베이지안 평균):
            단순 비율은 대화수가 작을 때 분산이 폭주한다. 이를 막기 위해:
                adjusted = (C·m + likes) / (C·1 + conversations)
              - m = 전체 후보의 평균 비율 (prior)
              - C = 신뢰도 상수 (quality_confidence_k)
            대화수가 C보다 훨씬 작으면 결과가 m에 가까워지고,
            C보다 훨씬 크면 그 작품 고유 비율에 수렴한다.

        ※ 반환값(0.006~0.014 수준)은 raw 값이며, 후보 간 격차가 매우
          작다. 실제 점수화는 _rerank 에서 Min-Max 정규화를 거친다.
        """
        meta = candidate["metadata"]
        conversations = max(0, int(meta.get("conversation_count", 0)))
        likes = max(0, int(meta.get("like_count", 0)))

        C = self.quality_confidence_k
        m = global_mean_ratio

        # 베이지안 평균. 분모는 항상 C(>0) 이상이라 0 division 위험 없음.
        adjusted = (C * m + likes) / (C * 1.0 + conversations)
        return max(0.0, min(1.0, adjusted))

    # ==================================================================
    # ④ 4축 재정렬
    # ==================================================================
    def _rerank(
        self, candidates: list[dict], accumulated_positives: list[str]
    ) -> list[dict]:
        """
        후보군을 4축 가중 합산으로 재정렬한다.
        최종점수 = 0.35·정규화유사도 + 0.30·태그매칭
                 + 0.20·정규화품질   + 0.15·최신성

        ★ 유사도와 품질은 '둘 다' 후보군 내 Min-Max 정규화한다.
          - 유사도: raw 값이 0.2~0.5 의 좁은 밴드 → 정규화로 펼침.
          - 품질: raw 비율이 0.006~0.014 의 더 좁은 밴드 → 정규화 필수.
            정규화 없이는 가중치 0.20 을 곱해도 기여분이 0.002 수준에
            그쳐 품질 축이 상수화(무력화)된다.
        품질 축은 임베딩과 독립적인 신호이므로, 한국어 임베딩의 뉘앙스
          한계로 유사도 축이 흔들려도 순위 안정성에 기여한다(리스크 분산).
        """
        # 0단계: 품질 보정 기준점(후보군 전체 평균 비율) 산출.
        global_mean = self._compute_global_mean_ratio(candidates)

        # 1단계: 각 후보의 raw 점수 3종 계산.
        #         (raw_similarity 는 vector_store 가 이미 넣어줬다.)
        for cand in candidates:
            cand["_recency"] = self._recency_score(
                cand["metadata"]["updated_at"]
            )
            cand["_tagmatch"] = self._tag_match_score(
                cand["metadata"].get("tags_normalized", ""),
                accumulated_positives,
            )
            cand["_quality_raw"] = self._quality_score(cand, global_mean)

        # 2단계: 후보군 내 Min/Max 추출 — 유사도와 품질 둘 다.
        sims = [c["raw_similarity"] for c in candidates]
        sim_min, sim_max = min(sims), max(sims)
        sim_range = sim_max - sim_min

        quals = [c["_quality_raw"] for c in candidates]
        qual_min, qual_max = min(quals), max(quals)
        qual_range = qual_max - qual_min

        # 3단계: 정규화 + 4축 가중 합산.
        scored = []
        for cand in candidates:
            raw_sim = cand["raw_similarity"]
            raw_qual = cand["_quality_raw"]

            # 유사도 정규화. 분모 0(전부 동일) 방어 → 중립값 0.5.
            if sim_range == 0:
                norm_sim = 0.5
            else:
                norm_sim = (raw_sim - sim_min) / sim_range

            # 품질 정규화. 동일하게 분모 0 방어 → 중립값 0.5.
            if qual_range == 0:
                norm_qual = 0.5
            else:
                norm_qual = (raw_qual - qual_min) / qual_range

            final = (
                self.w_sim * norm_sim
                + self.w_tag * cand["_tagmatch"]
                + self.w_qual * norm_qual
                + self.w_rec * cand["_recency"]
            )

            # 점수 내역 보존 → 발표 점수표 및 디버깅용.
            cand_scored = dict(cand)
            cand_scored["raw_similarity_r"] = round(raw_sim, 4)
            cand_scored["norm_similarity"] = round(norm_sim, 4)
            cand_scored["tagmatch_score"] = round(cand["_tagmatch"], 4)
            cand_scored["quality_raw"] = round(raw_qual, 6)
            cand_scored["quality_score"] = round(norm_qual, 4)
            cand_scored["recency_score"] = round(cand["_recency"], 4)
            cand_scored["final_score"] = round(final, 4)
            scored.append(cand_scored)

        scored.sort(key=lambda c: c["final_score"], reverse=True)
        return scored

    # ==================================================================
    # 발표용: 재정렬 점수표를 터미널에 줄 맞춰 출력
    # ==================================================================
    @staticmethod
    def print_ranking_table(ranking: list[dict]) -> None:
        """[raw유사, 정규화, 태그매칭, 품질, 최신성, 최종점수] 표 출력."""
        print("\n=== 📊 v3 재정렬 점수표 (4축: 유사도+태그+품질+최신성) ===")
        header = (
            f"{'캐릭터명':<24}{'raw유사':>9}{'정규화':>9}"
            f"{'태그매칭':>10}{'품질':>9}{'최신성':>9}{'최종점수':>11}"
        )
        print(header)
        print("-" * len(header))
        for r in ranking:
            name = r["metadata"]["name"]
            # 한글 폭 보정: 너무 길면 잘라서 표가 안 깨지게.
            disp = name if len(name) <= 22 else name[:21] + "…"
            print(
                f"{disp:<24}"
                f"{r['raw_similarity_r']:>9.4f}"
                f"{r['norm_similarity']:>9.4f}"
                f"{r['tagmatch_score']:>10.4f}"
                f"{r['quality_score']:>9.4f}"
                f"{r['recency_score']:>9.4f}"
                f"{r['final_score']:>11.4f}"
            )

    # ==================================================================
    # ⑤ 추천 멘트 생성 — 1위 캐릭터/세계관에 빙의 (단발)
    # ==================================================================
    def _generate_message(self, profile: dict, user_profile: dict) -> str:
        """
        선정된 캐릭터/세계관에 빙의해 Claude가 추천 대화를 생성한다.

        v2 이상: 수집 불가능한 description 대신, 안정적으로 수집 가능한
        name + tags_raw + tags_normalized + category 만으로 페르소나 구성.

        ★ 주의: 이 메서드의 캐릭터 파라미터명은 'profile' 이다.
          chat_with_character 의 'character_profile' 과 혼동하지 말 것.
          (붙여넣기 시 변수명 불일치로 NameError 가 나기 쉬운 지점.)
        """
        mood = user_profile.get("current_mood", "")
        system_prompt = (
            f"너는 '{profile['name']}'라는 AI 콘텐츠다. "
            f"장르: {profile.get('genre', '')}, "
            f"분위기 태그: {profile.get('tags_raw', '')}. "
            "이 작품의 화자가 되어 작품 특유의 말투와 분위기로 "
            "1인칭으로 말하라."
        )
        user_prompt = f"""[너의 프로필]
- 이름/세계관: {profile['name']}
- 카테고리: {profile.get('category', '')}
- 분위기 태그(원본): {profile.get('tags_raw', '')}
- 감성 키워드(정규화): {profile.get('tags_normalized', '')}

[사용자의 현재 상태]
- 정서: {mood}

[지시]
위 사용자에게, 왜 지금 너(이 작품)가 그에게 어울리는지를
2~3문장의 매력적인 대화체로, 작품의 분위기를 살려 직접 말을
건네듯 표현하라. 설명문이 아니라 작품 속 화자가 사용자를
초대하는 첫 마디처럼."""

        message = self.llm.messages.create(
            model=LLM_MODEL,
            max_tokens=400,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return message.content[0].text

    # ==================================================================
    # 공개 메서드: 누적 프로필 기반 추천 실행
    # ==================================================================
    def recommend(self, user_profile: dict, target_orientation: str) -> dict:
        """
        UserProfileManager 가 누적한 user_profile 을 입력받아 추천한다.

        user_profile 구조:
          {"accumulated_positives":[...], "accumulated_negatives":[...],
           "preferred_genre": ..., "current_mood": "..."}

        반환 dict:
          ranking       : 재정렬된 최종 후보(점수 내역 포함)
          winner        : 1위 캐릭터 전체 프로필
          message       : Claude 생성 추천 멘트
          fallback_used : 네거티브 배제 결과가 0건이라 폴백했는지 여부
        """
        # 검색 쿼리 텍스트: 누적 긍정 태그로 구성.
        # (current_mood 는 '명확한 선호도 표현' 같은 메타 표현이 섞여
        #  검색 노이즈가 될 수 있어 쿼리에서 제외. mood 는 멘트 생성에만 사용.)
        query_text = " ".join(user_profile["accumulated_positives"]).strip()
        if not query_text:
            # 긍정 태그가 비었으면 mood 라도 사용 (극단적 빈 입력 방어).
            query_text = user_profile.get("current_mood", "").strip() or "추천"

        # ① 검색: 고정 성향 + 누적 선호 장르 + 임계값 가드.
        candidates = self.store.search_characters(
            query_text=query_text,
            target_orientation=target_orientation,
            genre=user_profile.get("preferred_genre"),
            top_k=15,
            threshold_guard=self.threshold_guard,
        )
        if not candidates:
            raise ValueError(
                "후보가 0건입니다. 필터 조건 또는 DB 적재 상태를 확인하세요."
            )

        # ② 네거티브 스크리닝 (누적 기피 태그).
        screened = self._screen_negative(
            candidates, user_profile["accumulated_negatives"]
        )

        # ★ 엣지 케이스 방어: 배제 후 후보가 0건이면 배제 전 후보로 폴백.
        fallback_used = False
        if not screened:
            print("⚠️  기피 태그 배제 후 후보 0건 → 배제 전 후보로 폴백.")
            screened = candidates
            fallback_used = True

        # ③ 4축 랭킹.
        ranking = self._rerank(
            screened, user_profile["accumulated_positives"]
        )
        winner_id = ranking[0]["id"]
        if winner_id in self._profiles:
            winner_profile = self._profiles[winner_id]
        else:
            winner_profile = ranking[0]["metadata"]

        # ④ 멘트 생성.
        message = self._generate_message(winner_profile, user_profile)

        return {
            "ranking": ranking,
            "winner": winner_profile,
            "message": message,
            "fallback_used": fallback_used,
        }

    # ==================================================================
    # v3 추가: 1위 캐릭터와의 실시간 멀티턴 RAG 대화
    # ==================================================================
    def chat_with_character(
        self,
        character_profile: dict,
        chat_history: list[dict],
        user_message: str,
    ) -> str:
        """
        랭킹 1위 캐릭터에 빙의하여 유저와 실시간 연속 대화를 한다.

        v2까지는 추천 멘트 1회로 끝났으나, v3는 유저가 그 캐릭터와
        대화를 이어가는 챗봇 루프다.

        Context Augmentation:
          - character_profile: 1위 캐릭터의 페르소나(이름·태그·세계관)를
            system 프롬프트에 주입 → 캐릭터가 일관되게 유지된다.
          - chat_history: 지금까지의 전체 대화 기록을 messages 에 주입
            → Claude 가 맥락을 기억하고 연속성 있게 응답한다.

        character_profile : 캐릭터 전체 프로필 dict.
            (주의: _generate_message 의 'profile' 과 다른 파라미터명이다.)
        chat_history : 이전 대화 기록. [{"role":"user"/"assistant",
                       "content":"..."}, ...] 형식. 첫 턴이면 빈 리스트.
        user_message : 이번 턴 유저 발화.

        반환: 캐릭터의 응답 텍스트(str).
        """
        # 캐릭터 페르소나를 system 프롬프트로 고정.
        system_prompt = (
            f"너는 '{character_profile['name']}'라는 캐릭터/세계관이다.\n"
            f"장르: {character_profile.get('genre', '')}\n"
            f"분위기 태그: {character_profile.get('tags_raw', '')}\n"
            f"성격 키워드: {character_profile.get('tags_normalized', '')}\n\n"
            "위 설정에 100% 빙의하여, 이 캐릭터의 말투와 세계관을 일관되게 "
            "유지한 채 유저와 대화하라. 캐릭터를 절대 벗어나지 말고, "
            "메타적 설명 없이 캐릭터로서 자연스럽게 응답하라.\n"
            "★ 응답은 2~3문장으로 간결하게 하라. 첫째·둘째 같은 목록 나열 "
            "방식은 쓰지 말고, 캐릭터가 실제로 건넬 법한 짧은 대사로 답하라."
        )

        # 대화 기록 + 이번 턴 발화를 messages 로 조립.
        # chat_history 가 누적될수록 Claude 는 더 긴 맥락을 본다.
        messages = list(chat_history)  # 원본 보호를 위해 복사
        messages.append({"role": "user", "content": user_message})

        response = self.llm.messages.create(
            model=LLM_MODEL,
            max_tokens=600,  # 간결 지시를 줘도 문장 잘림 방지를 위한 여유
            system=system_prompt,
            messages=messages,
        )
        return response.content[0].text