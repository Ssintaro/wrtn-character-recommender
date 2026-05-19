"""scripts/recommend.py
고도화된 5단계 하이브리드 추천 파이프라인 전체를 원스톱으로 검증한다.
"""
from datetime import date

from dotenv import load_dotenv

from src.recommender.engine import CharacterRecommenderEngine

load_dotenv()  # OPENAI_API_KEY + ANTHROPIC_API_KEY 로드


def main():
    # 명세 지정: 유저 성향 프로필을 '남성(male)'으로 고정.
    user_orientation = "male"

    # 명세 지정 테스트 질문 (긍정 의도 + 부정 기피가 동시에 담긴 문장).
    user_query = (
        "요즘 마감 때문에 밤새서 피곤해. 이세계 전생해서 모험하는 "
        "시뮬레이션 같은 거 없나? 아, 연애 위주 취향인 로맨스는 절대 싫어"
    )

    # 엔진 생성. 기준일을 2026-05-19 로 고정 (발표 재현성 보장).
    engine = CharacterRecommenderEngine(
        weight_similarity=0.6,
        weight_recency=0.15,
        weight_category=0.25,
        reference_date=date(2026, 5, 19),
    )

    print(f"👤 유저 성향(고정): {user_orientation}")
    print(f"🙋 유저 질문: {user_query}\n")

    result = engine.recommend(user_query, target_orientation=user_orientation)

    # --- 재정렬 점수표 (발표 핵심 자료) ---
    engine.print_ranking_table(result["ranking"])

    if result["fallback_used"]:
        print("\n※ 기피 조건이 너무 엄격해 일부 완화하여 추천했습니다.")

    # --- 최종 추천 ---
    winner = result["winner"]
    print(f"\n🏆 최종 추천: {winner['name']}")
    print(
        f"   장르={winner['genre']} | 카테고리={winner['category']} | "
        f"업데이트={winner['updated_at']}"
    )

    print("\n=== 💬 Claude가 생성한 추천 멘트 ===")
    print(result["message"])


if __name__ == "__main__":
    main()