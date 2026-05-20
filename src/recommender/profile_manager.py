"""src/recommender/profile_manager.py

역할: Multi-turn 대화에서 유저의 취향을 누적 추출·유지하는 UserProfileManager.

v1과의 핵심 차이:
  v1은 매 질문을 독립적으로 처리하는 '단발성(stateless)' 구조였다.
  v2는 대화 턴을 거치며 유저 취향을 '축적(stateful)'한다.
  유저가 2턴에서 "로맨스 빼줘"라고 하면, 그 기피 의사가 UserProfile에
  영구 저장되어 이후 모든 턴의 추천에 계속 반영된다.
"""
import json

from anthropic import Anthropic

LLM_MODEL = "claude-haiku-4-5-20251001"

# tags_normalized 와 매칭될 정서/소재 표준 어휘.
# 인텐트 추출 시 LLM이 이 목록 밖 단어를 뱉으면 매칭이 무력화되므로,
# 프롬프트에 이 목록을 박고 코드에서도 화이트리스트로 거른다.
VALID_TAGS = [
    "위로", "힐링", "잔잔함", "공감", "설렘", "달달함", "애틋함", "다정함",
    "까칠함", "과묵함", "능글맞음", "티격태격", "코미디", "유쾌함", "집착",
    "얀데레", "긴장감", "어두움", "애증", "생존", "성장", "지적임", "전략",
    "헌신", "청춘", "액션", "판타지", "로맨스", "일상",
]
VALID_GENRES = ["roleplay_1on1", "simulation"]


class UserProfileManager:
    """대화 기반 유저 취향 프로필을 관리하는 클래스."""

    def __init__(self):
        self.llm = Anthropic()  # 환경변수 ANTHROPIC_API_KEY 자동 인식

        # ★ 세션 상태(Session State): 대화 전체에 걸쳐 유지되는 유저 프로필.
        #   이 객체가 v2의 '기억' 그 자체다. 턴이 바뀌어도 사라지지 않는다.
        self.profile = {
            "accumulated_positives": [],  # 누적 긍정 선호 태그
            "accumulated_negatives": [],  # 누적 기피 태그/장르
            "preferred_genre": None,      # 선호 장르 (최신 발화 우선)
            "current_mood": "",           # 현재 정서 상태 (최신 발화 우선)
        }

    # ------------------------------------------------------------------
    # 한 턴의 발화를 분석해 '이번 턴에 드러난 의도'를 추출
    # ------------------------------------------------------------------
    def _extract_turn_intent(self, user_message: str) -> dict:
        """
        유저의 한 턴 발화를 Claude에 던져, 이번 턴에서 드러난 의도를
        구조화 JSON으로 추출한다. (아직 누적 전, 순수 '이번 턴' 정보)

        반환: {"positives":[...], "negatives":[...],
               "genre": ...|null, "mood": "..."}
        """
        tag_list = ", ".join(VALID_TAGS)
        system_prompt = (
            "너는 유저 발화에서 콘텐츠 취향을 추출하는 파서다. 규칙:\n"
            f"1. positives/negatives 의 원소는 반드시 다음 목록 안의 "
            f"단어만 사용한다: [{tag_list}].\n"
            "   목록에 없는 단어는 의미가 가장 가까운 목록 내 단어로 바꾼다.\n"
            f"2. genre 는 'roleplay_1on1', 'simulation' 중 하나 또는 null.\n"
            "3. negatives 에는 유저가 '싫다/빼달라'고 명시한 것만 넣는다.\n"
            "4. mood 는 유저의 현재 정서를 짧은 한국어 구절로 (예: '지치고 피곤함').\n"
            "5. 출력은 오직 JSON 객체 하나. 설명·코드블록 금지."
        )
        user_prompt = (
            f'다음 발화를 분석해 JSON으로만 답하라.\n\n발화: "{user_message}"\n\n'
            '형식: {"positives": [...], "negatives": [...], '
            '"genre": ..., "mood": "..."}'
        )

        message = self.llm.messages.create(
            model=LLM_MODEL,
            max_tokens=300,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw_text = message.content[0].text.strip()

        # LLM이 ```json 펜스를 붙일 경우 방어적으로 제거.
        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`")
            if raw_text.lower().startswith("json"):
                raw_text = raw_text[4:]

        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            print("⚠️  턴 의도 파싱 실패 → 빈 의도로 처리.")
            parsed = {}

        return parsed

    # ------------------------------------------------------------------
    # 추출된 턴 의도를 누적 프로필에 '병합(Merge)'
    # ------------------------------------------------------------------
    def update(self, user_message: str) -> dict:
        """
        한 턴의 발화를 받아 프로필을 갱신하고, 갱신된 프로필을 반환한다.

        병합 규칙:
          - positives/negatives: 누적(append)한다. 단 중복은 제거.
            → 과거 턴의 취향이 사라지지 않고 쌓인다.
          - genre/mood: 최신 발화 값으로 덮어쓴다(overwrite).
            → 장르나 기분은 '현재 상태'가 중요하므로 최신값 우선.
          - 모순 해소: 이번 턴 negative 에 들어온 태그가 기존 positives 에
            있었다면 positives 에서 제거한다. ("역시 로맨스 빼줘" 케이스)
        """
        turn = self._extract_turn_intent(user_message)

        # --- 화이트리스트 검증 (LLM 출력을 그대로 믿지 않는다) ---
        def _clean_tags(items) -> list[str]:
            if not isinstance(items, list):
                return []
            return [t for t in items if t in VALID_TAGS]

        turn_pos = _clean_tags(turn.get("positives"))
        turn_neg = _clean_tags(turn.get("negatives"))

        # --- positives 누적 (집합으로 중복 제거 후 리스트 복원) ---
        pos_set = set(self.profile["accumulated_positives"])
        pos_set.update(turn_pos)

        # --- negatives 누적 ---
        neg_set = set(self.profile["accumulated_negatives"])
        neg_set.update(turn_neg)

        # --- 모순 해소: 새로 기피한 태그는 긍정 목록에서 제거 ---
        pos_set -= neg_set

        self.profile["accumulated_positives"] = sorted(pos_set)
        self.profile["accumulated_negatives"] = sorted(neg_set)

        # --- genre: 이번 턴에 유효 값이 있으면 덮어쓰기 ---
        turn_genre = turn.get("genre")
        if turn_genre in VALID_GENRES:
            self.profile["preferred_genre"] = turn_genre

        # --- mood: 누적 방식 ---
        # ★ v2.1 수정: mood 를 덮어쓰지 않고 누적한다.
        #   동일 세션은 '같은 날, 같은 맥락의 연속 대화'이므로,
        #   1턴의 '피곤함'이 2턴에서 사라지면 안 된다. 새 턴의 정서가
        #   기존과 다르면 둘을 함께 보존해, 멘트 생성이 유저의 누적된
        #   정서 맥락 전체를 반영하게 한다.
        #   (세션이 바뀌는 '다른 날 재방문'은 새 매니저 인스턴스 생성으로
        #    자연히 리셋되므로, 세션 내에서는 누적이 올바른 동작이다.)
        turn_mood = turn.get("mood")
        if isinstance(turn_mood, str) and turn_mood.strip():
            new_mood = turn_mood.strip()
            existing = self.profile["current_mood"].strip()
            if not existing:
                # 첫 mood → 그대로 설정.
                self.profile["current_mood"] = new_mood
            elif new_mood in existing:
                # 이미 같은 정서가 기록돼 있으면 중복 추가하지 않는다.
                pass
            else:
                # 다른 정서 → 기존 정서에 이어붙여 누적.
                self.profile["current_mood"] = f"{existing}, {new_mood}"

        return self.profile

    def get_profile(self) -> dict:
        """현재 누적된 유저 프로필을 반환한다."""
        return self.profile