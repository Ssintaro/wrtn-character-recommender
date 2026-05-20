"""scripts/run_evaluation.py
LLM-as-a-Judge 평가 실행: 테스트 데이터셋 전체를 추천 엔진에 돌리고,
판정자(GPT)로 채점하여 평균 점수 성적표를 출력한다.
"""
from datetime import date

from dotenv import load_dotenv

from src.recommender.profile_manager import UserProfileManager
from src.recommender.engine import CharacterRecommenderEngine
from src.evaluation.harness import RecommendationJudge

load_dotenv()  # OPENAI_API_KEY + ANTHROPIC_API_KEY 로드


# ======================================================================
# 정밀 실험 데이터셋 — 추천 엔진의 한계를 다각도로 검증하는 7개 시나리오.
#   각 시나리오는 turns(유저 발화 리스트)로 구성. 발화가 2개면 멀티턴.
# ======================================================================
EVAL_DATASET = [
    {
        "id": "A-1",
        "type": "순수 감성/위로형",
        "orientation": "female",
        "turns": [
            "요즘 인간관계 때문에 너무 피곤하고 마음이 지쳐... "
            "잔잔하게 내 이야기 들어주고 위로해 줄 친구 없을까?"
        ],
    },
    {
        "id": "B-1",
        "type": "장르+하드 기피 조건형 (v2/v3 검증)",
        "orientation": "female",
        "turns": [
            "달달하고 설레는 로맨스풍 시뮬레이션 추천해줘.",
            "아, 근데 너무 오글거리는 집착이나 얀데레 성격은 무조건 제외해줘.",
        ],
    },
    {
        "id": "C-1",
        "type": "서사 기반 액션/생존형 (v1 노이즈 방어 검증)",
        "orientation": "male",
        "turns": [
            "아포칼립스나 던전 같은 극악 난이도 세계관에서 주인공이랑 "
            "같이 구르는 빡센 생존 시뮬레이션 물이 필요해."
        ],
    },
    {
        "id": "D-1",
        "type": "모호한 입력 (임계값 가드/폴백 검증)",
        "orientation": "male",
        "turns": ["그냥 재밌는 거."],
    },
    {
        "id": "A-2",
        "type": "순수 감성/위로형 (변형)",
        "orientation": "male",
        "turns": [
            "하루 종일 일에 치여서 너무 외롭다. 따뜻하게 말 걸어주는 "
            "다정한 상대가 필요해."
        ],
    },
    {
        "id": "B-2",
        "type": "장르+기피 조건형 (변형)",
        "orientation": "female",
        "turns": [
            "두근거리는 학원물 로맨스 보고 싶어.",
            "근데 어둡고 우울한 분위기는 빼줘. 밝은 게 좋아.",
        ],
    },
    {
        "id": "C-2",
        "type": "서사 기반 액션/생존형 (변형)",
        "orientation": "male",
        "turns": [
            "치열한 전투랑 전략이 살아있는 무협 세계관에서 살아남는 "
            "이야기를 하고 싶어."
        ],
    },
]


def run_one_scenario(
    scenario: dict,
    engine: CharacterRecommenderEngine,
    judge: RecommendationJudge,
) -> dict:
    """
    시나리오 1건을 실행한다.
    멀티턴이면 모든 턴을 프로필에 누적한 뒤, 마지막 상태로 추천·평가한다.
    반환: {"id","type","scores":{...}}
    """
    # 매 시나리오마다 프로필 매니저를 새로 생성 → 시나리오 간 취향 격리.
    profile_mgr = UserProfileManager()

    profile = None
    for turn in scenario["turns"]:
        profile = profile_mgr.update(turn)

    # 누적된 최종 프로필로 추천 실행.
    result = engine.recommend(
        profile, target_orientation=scenario["orientation"]
    )

    # 추천 결과를 판정자에게 넘겨 채점.
    # 마지막 턴 발화를 '유저 질문' 대표값으로 사용.
    last_query = scenario["turns"][-1]
    # print(f"  [진단] judge.evaluate 호출 직전 — winner='{result['winner'].get('name','?')}'")
    verdict = judge.evaluate(
        user_query=last_query,
        character_profile=result["winner"],
        recommendation_message=result["message"],
    )
    # print(f"  [진단] verdict 반환됨 — failed={verdict.get('failed','키없음')}, scores={verdict['scores']}")

    return {
        "id": scenario["id"],
        "type": scenario["type"],
        "winner": result["winner"].get("name", ""),
        "scores": verdict["scores"],
    }


def main():
    print("=" * 70)
    print("  LLM-as-a-Judge 추천 품질 자동 평가  (생성: Claude / 평가: GPT)")
    print("=" * 70)

    engine = CharacterRecommenderEngine(reference_date=date(2026, 5, 19))
    judge = RecommendationJudge()

    results = []
    for scenario in EVAL_DATASET:
        print(f"\n▶ 시나리오 {scenario['id']} ({scenario['type']}) 평가 중...")
        outcome = run_one_scenario(scenario, engine, judge)
        results.append(outcome)
        s = outcome["scores"]
        print(
            f"  추천='{outcome['winner']}' | "
            f"적합성 {s['relevance']} / 설명력 {s['explainability']} / "
            f"충실성 {s['faithfulness']}"
        )

    # ===== 평가 성적표 =====
    print("\n" + "=" * 70)
    print("  📋 평가 성적표")
    print("=" * 70)
    header = (
        f"{'시나리오':<10}{'유형':<26}"
        f"{'적합성':>8}{'설명력':>8}{'충실성':>8}"
    )
    print(header)
    print("-" * len(header))

    total_rel = total_exp = total_fai = 0
    for r in results:
        s = r["scores"]
        total_rel += s["relevance"]
        total_exp += s["explainability"]
        total_fai += s["faithfulness"]
        type_disp = r["type"] if len(r["type"]) <= 24 else r["type"][:23] + "…"
        print(
            f"{r['id']:<10}{type_disp:<26}"
            f"{s['relevance']:>8}{s['explainability']:>8}{s['faithfulness']:>8}"
        )

    n = len(results)
    print("-" * len(header))
    print(
        f"{'평균':<10}{'':<26}"
        f"{total_rel / n:>8.2f}{total_exp / n:>8.2f}{total_fai / n:>8.2f}"
    )
    overall = (total_rel + total_exp + total_fai) / (n * 3)
    print(f"\n  ⭐ 종합 평균 점수: {overall:.2f} / 5.00")
    print("=" * 70)


if __name__ == "__main__":
    main()