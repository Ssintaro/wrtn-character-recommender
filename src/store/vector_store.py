"""src/store/vector_store.py

역할: 스키마 v2 캐릭터 데이터를 ChromaDB에 적재하고, 하드 필터를 적용해
     유사 캐릭터 후보군을 검색한다.

스키마 v2 메타데이터 필드:
  character_id, name, creator_name, creator_followers(int),
  genre, category, target_orientation, conversation_count(int),
  like_count(int), comment_count(int), updated_at, tags_raw, tags_normalized
"""
from pathlib import Path

import chromadb

from src.embedding.embedder import OpenAIEmbedder


class CharacterVectorStore:
    """ChromaDB 기반 캐릭터 벡터 저장소 (스키마 v2)."""

    def __init__(
        self,
        persist_dir: str = "chroma_db",
        collection_name: str = "crack_characters_v2",
    ):
        # 임베딩 번역기.
        self.embedder = OpenAIEmbedder()

        # PersistentClient: 데이터를 디스크 폴더에 영구 저장.
        abs_path = str(Path(persist_dir).resolve())
        self.client = chromadb.PersistentClient(path=abs_path)

        # 컬렉션. space=cosine: 의미 유사도엔 코사인이 표준.
        # 스키마가 바뀌었으므로 컬렉션 이름을 v2로 둬서 구버전과 분리한다.
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # 내부 헬퍼: 캐릭터 1명 → 임베딩용 자연어 문장
    # ------------------------------------------------------------------
    @staticmethod
    def _build_embedding_text(character: dict) -> str:
        """
        이름 + 날것 태그 + 정규화 태그를 결합해 임베딩 입력 문장을 만든다.

        ★ tags_raw(날것)를 일부러 그대로 넣는다:
          무한 태그의 다양성은 임베딩에겐 노이즈가 아니라 풍부한 의미 신호다.
          '#밤샘공감' 같은 희귀 태그도 모델이 알아서 '위로'에 가깝게 매핑한다.
        """
        return (
            f"{character['name']}. "
            f"{character.get('tags_raw', '').replace(',', ' ')}. "
            f"{character.get('tags_normalized', '').replace(',', ' ')}"
        ).strip()

    # ------------------------------------------------------------------
    # 내부 헬퍼: 캐릭터 1명 → ChromaDB 메타데이터 dict (스키마 v2)
    # ------------------------------------------------------------------
    @staticmethod
    def _build_metadata(character: dict) -> dict:
        """
        스키마 v2 전체 필드를 ChromaDB 메타데이터로 가공한다.

        ★ 주의 2가지:
          (1) ChromaDB 메타데이터 값은 str/int/float/bool 만 허용.
          (2) 숫자(count, followers)는 반드시 int 로 저장한다.
              문자열로 넣으면 향후 숫자 비교 필터($gt 등)가 동작하지 않는다.
        """
        return {
            "character_id": str(character["character_id"]),
            "name": str(character["name"]),
            "creator_name": str(character.get("creator_name", "")),
            # 인기 편향 제어용 수치 → int 강제 변환.
            "creator_followers": int(character.get("creator_followers", 0)),
            # 하드 필터용 분류 필드들.
            "genre": str(character.get("genre", "")),
            "category": str(character.get("category", "")),
            "target_orientation": str(character.get("target_orientation", "")),
            # 인기/활성도 메트릭.
            "conversation_count": int(character.get("conversation_count", 0)),
            "like_count": int(character.get("like_count", 0)),
            "comment_count": int(character.get("comment_count", 0)),
            # 최신성 계산용. "YYYY-MM-DD" 문자열.
            "updated_at": str(character.get("updated_at", "")),
            # 태그 2종 모두 보존.
            "tags_raw": str(character.get("tags_raw", "")),
            "tags_normalized": str(character.get("tags_normalized", "")),
        }

    # ------------------------------------------------------------------
    # 적재: 캐릭터 리스트를 임베딩 + 메타데이터와 함께 저장
    # ------------------------------------------------------------------
    def add_characters(self, character_list: list[dict]) -> int:
        """전달받은 캐릭터 리스트를 ChromaDB에 적재. 반환: 적재 수."""
        if not character_list:
            print("⚠️  추가할 캐릭터가 없습니다.")
            return 0

        # ChromaDB의 upsert 는 '컬럼 단위 리스트'를 요구한다.
        ids = [str(c["character_id"]) for c in character_list]
        documents = [self._build_embedding_text(c) for c in character_list]
        metadatas = [self._build_metadata(c) for c in character_list]

        # 결합 문장 전체를 한 번의 배치 호출로 벡터화.
        print(f"🧮 {len(documents)}개 캐릭터를 임베딩하는 중...")
        embeddings = self.embedder.embed_texts(documents)

        # upsert: 같은 id가 있으면 갱신, 없으면 추가 (재실행 안전).
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )
        print(f"✅ {len(ids)}개 캐릭터를 ChromaDB에 저장 완료.")
        return len(ids)

    # ------------------------------------------------------------------
    # 검색: 고정 성향 필터 + 동적 장르 필터 적용 후 top_k 반환
    # ------------------------------------------------------------------
    def search_characters(
        self,
        query_text: str,
        target_orientation: str,
        genre: str | None = None,
        top_k: int = 15,
    ) -> list[dict]:
        """
        유저 질문과 유사한 캐릭터 후보군을 하드 필터를 걸어 추출한다.

        target_orientation: 앱 진입 시 고정된 유저 성향 ("male"/"female").
            ★ 핵심: 유저 성향과 '정확히 일치'하는 캐릭터뿐 아니라,
              'unspecified'(공용) 캐릭터도 함께 통과시킨다.
              공용 캐릭터를 버리면 무협·2차창작 등이 통째로 누락된다.
              → $in 연산자로 [유저성향, unspecified] 둘 다 허용.
        genre: 인텐트 파싱으로 추론된 동적 장르. None 이면 장르 제한 없음.
        top_k: 넉넉히 추출 (이후 네거티브 배제로 줄어들 것을 감안).

        반환: [{"id","document","metadata","distance"}, ...]
        """
        # 1) 유저 질문을 캐릭터와 같은 좌표계의 벡터로 변환.
        query_vector = self.embedder.embed_query(query_text)

        # 2) where 필터 구성.
        #    조건이 2개 이상이면 ChromaDB는 $and 로 묶어야 한다.
        conditions = []

        # (a) 고정 성향 필터: 유저 성향 + 공용(unspecified) 허용.
        conditions.append(
            {"target_orientation": {"$in": [target_orientation, "unspecified"]}}
        )

        # (b) 동적 장르 필터: 파싱으로 장르가 정해진 경우에만 추가.
        if genre:
            conditions.append({"genre": {"$eq": genre}})

        # 조건이 1개면 그대로, 2개 이상이면 $and 로 결합.
        if len(conditions) == 1:
            where_filter = conditions[0]
        else:
            where_filter = {"$and": conditions}

        # 3) 유사도 검색 실행.
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        # 4) ChromaDB 응답은 '리스트의 리스트'. 질문 1개이므로 [0]을 꺼낸다.
        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        # 5) 다루기 쉬운 dict 리스트로 재조립.
        hits = []
        for i in range(len(ids)):
            hits.append(
                {
                    "id": ids[i],
                    "document": docs[i],
                    "metadata": metas[i],
                    "distance": dists[i],
                }
            )
        return hits

    def count(self) -> int:
        """현재 저장된 캐릭터 수 (디버깅용)."""
        return self.collection.count()