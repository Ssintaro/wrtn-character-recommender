"""scripts/main_pipeline_v3.py
v3 통합 시연: 크롤링 → 정규화 적재 → 추천 → 실시간 RAG 대화.
"""
from datetime import date

from dotenv import load_dotenv

from src.crawler.collector import DynamicCharacterCrawler
from src.store.vector_store import CharacterVectorStore
from src.recommender.profile_manager import UserProfileManager
from src.recommender.engine import CharacterRecommenderEngine

load_dotenv()  # OPENAI_API_KEY + ANTHROPIC_API_KEY 로드


def main():
    user_orientation = "male"

    # ===== 1단계: 크롤러 가동 — 외부 캐릭터 데이터 동적 수집 =====
    print("\n" + "=" * 60)
    print("【 1단계 】 동적 웹 크롤링")
    print("=" * 60)
    crawler = DynamicCharacterCrawler()  # 타깃 미지정 → 로컬 가상 페이지
    raw_data = crawler.crawl()

    # ===== 2단계: 클렌징 + ChromaDB 증분 적재 =====
    print("=" * 60)
    print("【 2단계 】 데이터 정규화 및 멱등 적재")
    print("=" * 60)
    store = CharacterVectorStore()
    store.upsert_crawled_characters(raw_data)
    print(f"📦 현재 DB 총 캐릭터 수: {store.count()}\n")

    # ===== 3단계: 유저 프로파일링 + 추천 =====
    print("=" * 60)
    print("【 3단계 】 Multi-turn 프로파일링 및 추천")
    print("=" * 60)
    profile_mgr = UserProfileManager()
    engine = CharacterRecommenderEngine(reference_date=date(2026, 5, 19))

    turn1 = (
        "요즘 너무 지쳐서 빡센 생존 액션 시뮬레이션이 땡겨. "
        "던전 같은 거 들어가서 살아남는 거."
    )
    print(f"🗣️  유저: {turn1}")
    profile = profile_mgr.update(turn1)
    print(f"  🧾 프로필: 긍정={profile['accumulated_positives']}, "
          f"장르={profile['preferred_genre']}\n")

    result = engine.recommend(profile, target_orientation=user_orientation)
    engine.print_ranking_table(result["ranking"])

    winner = result["winner"]
    print(f"\n🏆 최종 추천: {winner['name']}")

    # ===== 4단계: 1위 캐릭터와 실시간 RAG 대화 루프 =====
    print("\n" + "=" * 60)
    print("【 4단계 】 1위 캐릭터와 실시간 RAG 대화")
    print("=" * 60)

    # 대화 기록을 메모리에 유지. 매 턴 user/assistant 쌍이 누적된다.
    chat_history = []
    user_turns = [
        "안녕? 반가워. 거긴 상황이 어때?",
        "위험해 보이는데... 내가 살아남으려면 뭘 먼저 해야 해?",
    ]

    for i, user_msg in enumerate(user_turns, start=1):
        print(f"\n🙋 [대화 {i}] 유저: {user_msg}")
        reply = engine.chat_with_character(
            character_profile=winner,
            chat_history=chat_history,
            user_message=user_msg,
        )
        print(f"🎭 [{winner['name']}]: {reply}")

        # 이번 턴을 기록에 누적 → 다음 턴이 맥락을 이어받는다.
        chat_history.append({"role": "user", "content": user_msg})
        chat_history.append({"role": "assistant", "content": reply})

    print("\n" + "=" * 60)
    print("✅ v3 전체 파이프라인 시연 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()