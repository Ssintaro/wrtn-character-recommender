"""scripts/build_index.py
characters.json(50개, 스키마 v2)을 읽어 ChromaDB에 적재하고 검색을 테스트한다.
"""
import json
from pathlib import Path

from dotenv import load_dotenv

from src.store.vector_store import CharacterVectorStore

load_dotenv()  # .env 의 OPENAI_API_KEY 를 환경변수로 로드

# 1) 캐릭터 데이터 로드
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "mock" / "characters.json"
with open(DATA_PATH, encoding="utf-8") as f:
    characters = json.load(f)

# 2) 저장소 준비 + 적재
store = CharacterVectorStore()
store.add_characters(characters)
print(f"📦 현재 DB에 저장된 캐릭터 수: {store.count()}\n")

# 3) 검색 테스트 — 스키마 v2 메서드(search_characters)에 맞춤.
#    target_orientation 은 필수 인자다. 여기선 'male' 로 테스트.
#    (성향 male + 공용 unspecified 캐릭터가 후보로 잡힌다.)
print("=== 검색 테스트: '일에 지쳐서 조용히 위로받고 싶어' (성향=male) ===")
hits = store.search_characters(
    query_text="일에 지쳐서 조용히 위로받고 싶어",
    target_orientation="male",
    genre=None,   # 장르 제한 없음
    top_k=3,
)
for hit in hits:
    m = hit["metadata"]
    # 스키마 v2: 'tags' → 'tags_normalized' 로 변경.
    print(
        f"  - {m['name']} (거리 {hit['distance']:.4f}) "
        f"| 장르: {m['genre']} | 태그: {m['tags_normalized']}"
    )