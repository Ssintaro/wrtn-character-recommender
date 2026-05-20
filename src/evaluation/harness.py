"""src/evaluation/harness.py

역할: LLM-as-a-Judge 기반 추천 품질 자동 평가 하니스.

핵심 설계:
  - 생성은 Claude, 평가는 OpenAI GPT 가 담당하는 '교차 검증' 구도.
    동일 모델이 자기 출력을 채점할 때 생기는 자기편향(self-preference
    bias)을 구조적으로 차단한다.
  - 판정자는 점수를 먼저 내지 않고, 각 지표의 채점 근거(reasoning)를
    CoT(Chain-of-Thought)로 먼저 서술한 뒤 점수를 매긴다.
  - RAGAS 라이브러리에 의존하지 않고, 그 평가 철학(검색·생성 품질의
    분리 측정)을 프롬프트로 자체 내재화한다.

★ Structured Outputs 도입 (JSON 구조 강제):
  이전에는 response_format={"type":"json_object"} 만 사용했다. 이는
  '유효한 JSON 으로 답하라'는 지시일 뿐, '이런 구조로 답하라'는 강제가
  아니다. 그 결과 판정자(LLM)가 호출마다 JSON 중첩 구조를 제멋대로
  바꾸어(scores 를 최상위에 두거나, reasoning 안에 넣거나, 아예 누락),
  점수 추출이 비결정적으로 실패했다.
  → response_format 에 명시적 JSON 스키마(json_schema)를 박는다.
    OpenAI 가 모델 출력을 스키마에 강제로 맞추므로, 판정자가 키 이름·
    중첩 구조를 어기는 것이 구조적으로 불가능해진다.
"""
import json

from openai import OpenAI

# ★ 판정자 모델. (2026-05 기준 developers.openai.com/api/docs/models 에서
#   gpt-5.4-mini 유효성 확인 완료.) OpenAI 모델명은 갱신이 잦으므로,
#   추후 호출 실패 시 이 상수부터 콘솔과 대조할 것.
JUDGE_MODEL = "gpt-5.4-mini"


# ======================================================================
# 판정자 응답 JSON 스키마
#   OpenAI Structured Outputs 에 주입되어, 판정자 출력이 반드시 이
#   구조를 따르도록 강제한다. additionalProperties=false 와 required 로
#   '정확히 이 키들, 이 중첩'만 허용한다.
# ======================================================================
EVALUATION_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "object",
            "description": "각 지표의 채점 근거를 CoT 로 먼저 서술한다.",
            "properties": {
                "relevance": {
                    "type": "string",
                    "description": "적합성 채점 근거 서술",
                },
                "explainability": {
                    "type": "string",
                    "description": "설명력 채점 근거 서술",
                },
                "faithfulness": {
                    "type": "string",
                    "description": "사실 충실성 채점 근거 서술",
                },
            },
            "required": ["relevance", "explainability", "faithfulness"],
            "additionalProperties": False,
        },
        "scores": {
            "type": "object",
            "description": "reasoning 을 근거로 도출한 1~5 정수 점수.",
            "properties": {
                "relevance": {
                    "type": "integer",
                    "description": "적합성 점수 (1~5)",
                },
                "explainability": {
                    "type": "integer",
                    "description": "설명력 점수 (1~5)",
                },
                "faithfulness": {
                    "type": "integer",
                    "description": "사실 충실성 점수 (1~5)",
                },
            },
            "required": ["relevance", "explainability", "faithfulness"],
            "additionalProperties": False,
        },
    },
    # 최상위에 reasoning 과 scores 가 '형제'로 반드시 존재해야 한다.
    "required": ["reasoning", "scores"],
    "additionalProperties": False,
}


class RecommendationJudge:
    """OpenAI GPT 기반 추천 품질 판정자."""

    def __init__(self):
        # OpenAI() 는 환경변수 OPENAI_API_KEY 를 자동으로 읽는다.
        self.client = OpenAI()

    # ------------------------------------------------------------------
    # 판정자 system 프롬프트 구성
    # ------------------------------------------------------------------
    @staticmethod
    def _build_system_prompt() -> str:
        """
        판정자에게 평가 기준·CoT 절차·모범 예시를 주입한다.

        ※ 출력 JSON 구조는 Structured Outputs 스키마가 강제하므로,
          프롬프트에서는 형식보다 '채점 기준'과 'CoT 절차'에 집중한다.
        """
        return """너는 AI 캐릭터 추천 시스템의 품질을 평가하는 엄정한 심사관이다.
주어진 [유저 질문], [추천된 캐릭터 프로필], [생성된 추천 멘트]를 보고
아래 3개 지표를 1~5점 척도로 채점한다.

[평가 지표]
1. relevance(적합성): 추천된 캐릭터가 유저의 의도·상황·기피조건에
   부합하는가? 유저가 원한 장르·정서와 캐릭터가 맞는지, 유저가 싫다고
   한 요소가 섞이지 않았는지를 본다.
2. explainability(설명력): 추천 멘트가 '왜 이 캐릭터인지'에 대한
   납득 가능한 근거를 충분히 제시하는가?
3. faithfulness(사실 충실성): 추천 멘트의 내용이 캐릭터 프로필(태그·
   장르·카테고리)에 실제로 근거하는가? 프로필에 없는 사실을 지어내지
   않았는가?

[채점 절차 — 반드시 이 순서를 지킬 것]
1) 먼저 reasoning 필드에 각 지표의 채점 근거를 구체적으로 서술한다.
   점수를 먼저 정하지 말고, 근거를 서술한 결과로서 점수가 도출되게 한다.
2) 그 다음 scores 필드에 각 지표의 1~5 정수 점수를 매긴다.
   reasoning 의 결론과 scores 의 점수는 논리적으로 일치해야 한다.

[채점 척도]
5=매우 우수, 4=우수, 3=보통, 2=미흡, 1=부적합.

[모범 채점 예시]
- 유저가 '위로받고 싶다'고 했는데 추천 캐릭터가 '까칠한 전투광'이고
  멘트도 위로와 무관하다면 → relevance 는 1~2점.
- 추천 캐릭터 태그에 '위로,힐링'이 있고 멘트가 그 점을 짚어 위로를
  약속한다면 → relevance 4~5, faithfulness 4~5."""

    # ------------------------------------------------------------------
    # 폴백 결과 (평가 실패 시 반환할 0점 dict)
    # ------------------------------------------------------------------
    @staticmethod
    def _fallback_result(reason: str) -> dict:
        """평가 실패 시 0점 결과를 반환. reason 에 실패 사유를 기록."""
        return {
            "reasoning": {
                "relevance": f"[평가 실패] {reason}",
                "explainability": f"[평가 실패] {reason}",
                "faithfulness": f"[평가 실패] {reason}",
            },
            "scores": {
                "relevance": 0,
                "explainability": 0,
                "faithfulness": 0,
            },
            # 실패 여부 플래그. 성적표 집계 시 0점을 '측정 실패'로
            # 구분해 평균에서 제외할 수 있게 한다.
            "failed": True,
        }

    # ------------------------------------------------------------------
    # 단일 추천 결과를 평가
    # ------------------------------------------------------------------
    def evaluate(
        self,
        user_query: str,
        character_profile: dict,
        recommendation_message: str,
    ) -> dict:
        """
        하나의 추천 결과(질문 + 추천 캐릭터 + 멘트)를 평가한다.

        반환:
          성공 시 {"reasoning":{...}, "scores":{...}, "failed": False}
          실패 시 {"reasoning":{...}, "scores":{0,0,0}, "failed": True}

        Structured Outputs 가 JSON 구조를 강제하므로, 정상 호출에서는
        scores 객체가 항상 최상위에 존재한다. 따라서 '점수 위치를
        찾아 헤매는' 로직이 불필요하다.
        """
        # 판정자에게 줄 평가 대상 정보를 구성.
        eval_input = f"""[유저 질문]
{user_query}

[추천된 캐릭터 프로필]
- 이름: {character_profile.get('name', '')}
- 장르: {character_profile.get('genre', '')}
- 카테고리: {character_profile.get('category', '')}
- 태그(원본): {character_profile.get('tags_raw', '')}
- 감성 키워드: {character_profile.get('tags_normalized', '')}

[생성된 추천 멘트]
{recommendation_message}"""

        # raw_text 를 미리 선언 → except 블록에서도 안전하게 참조 가능.
        raw_text = ""

        try:
            # --- 판정자(OpenAI) 호출 ---
            # response_format 에 json_schema 를 주입해 출력 구조를 강제.
            # strict=True 로 스키마를 엄격히 적용한다.
            response = self.client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": self._build_system_prompt(),
                    },
                    {"role": "user", "content": eval_input},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "recommendation_evaluation",
                        "strict": True,
                        "schema": EVALUATION_SCHEMA,
                    },
                },
            )

            # --- 거부(refusal) 응답 방어 ---
            # 모델이 안전상의 이유 등으로 평가를 거부하면 refusal 필드가 찬다.
            message = response.choices[0].message
            if getattr(message, "refusal", None):
                print(f"⚠️  평가 실패: 판정자가 평가를 거부함 — {message.refusal}")
                return self._fallback_result(f"판정자 거부: {message.refusal}")

            raw_text = (message.content or "").strip()

            # --- 빈 응답 방어 ---
            if not raw_text:
                print("⚠️  평가 실패: 판정자 응답 본문이 비어 있음.")
                print(f"   (finish_reason: "
                      f"{response.choices[0].finish_reason})")
                return self._fallback_result("판정자 응답이 비어 있음")

            # --- JSON 파싱 ---
            # Structured Outputs 가 구조를 보장하므로, 정상 응답이면
            # 항상 {"reasoning":{...}, "scores":{...}} 형태다.
            parsed = json.loads(raw_text)

            # [진단] 판정자가 실제로 보낸 JSON 구조를 출력.
            #   Structured Outputs 적용 후에도 구조가 흔들리는지 확인용.
            # print(f"  [진단] 판정자 JSON 구조: {parsed}")

            # --- 점수 추출 ---
            # 스키마가 scores 의 위치·키·타입을 강제하므로, 최상위에서
            # 바로 꺼낸다. 그래도 만일을 대비해 dict 타입은 확인한다.
            scores = parsed.get("scores")
            if not isinstance(scores, dict):
                print("⚠️  평가 실패: 응답에 scores 객체가 없음 "
                      "(스키마 강제가 적용되지 않았을 수 있음).")
                print(f"   판정자 JSON 구조: {parsed}")
                return self._fallback_result("scores 객체 부재")

            # --- 점수 정제: 정수 변환 + 1~5 범위 clamp ---
            for key in ("relevance", "explainability", "faithfulness"):
                val = int(scores.get(key, 0))
                scores[key] = max(0, min(5, val))
            parsed["scores"] = scores
            parsed["failed"] = False
            return parsed

        except json.JSONDecodeError as e:
            # 판정자가 JSON 이 아닌 텍스트를 반환한 경우 (극히 드묾).
            print(f"⚠️  평가 실패 (JSON 파싱 오류): {e}")
            print(f"   판정자 원본 응답(앞 300자): {raw_text[:300]!r}")
            return self._fallback_result(f"JSON 파싱 오류: {e}")

        except Exception as e:
            # OpenAI API 오류(모델명·파라미터·인증·rate limit 등),
            # 그 외 모든 예외를 포착해 원인을 노출한다.
            print(f"⚠️  평가 실패 ({type(e).__name__}): {e}")
            if raw_text:
                print(f"   판정자 원본 응답(앞 300자): {raw_text[:300]!r}")
            return self._fallback_result(f"{type(e).__name__}: {e}")