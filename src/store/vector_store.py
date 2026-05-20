"""src/store/vector_store.py  (v2)

v2 변경점:
  1. _build_embedding_text: description(서사 설명) 필드를 결합해
     '전생·모험·고립' 같은 서사 신호가 벡터에 반영되게 한다.
  2. search_characters: 절대적 유사도 임계값(threshold_guard)을 추가해
     노이즈 수준의 후보를 정규화 이전에 탈락시킨다.
"""
from pathlib import Path

import chromadb

from src.embedding.embedder import OpenAIEmbedder


class CharacterVectorStore:
    """ChromaDB 기반 캐릭터 벡터 저장소 (스키마 v2 / 엔진 v2)."""

    def __init__(
        self,
        persist_dir: str = "chroma_db",
        collection_name: str = "crack_characters_v2",
    ):
        self.embedder = OpenAIEmbedder()
        abs_path = str(Path(persist_dir).resolve())
        self.client = chromadb.PersistentClient(path=abs_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # ★ v2 개조: 임베딩 텍스트에 description 결합
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # ★ v2 개조: 장르별 임베딩 텍스트 분기
    # ------------------------------------------------------------------
    @staticmethod
    def _build_embedding_text(character: dict) -> str:
        """
        임베딩 입력 문장을 '장르에 따라 다르게' 구성한다.

        설계 근거 (v2 보고서 기재용):
          뤼튼 크랙의 두 장르는 제목의 정보 성격이 근본적으로 다르다.
          - simulation: 제목 자체가 세계관·서사의 요약이다.
            (예: "포스트 아포칼립스 : 마지막 분대" → 장르·상황이 제목에 노출)
            따라서 제목을 임베딩의 핵심 신호로 충분히 활용한다.
          - roleplay_1on1: 제목이 대개 캐릭터의 '이름'이라 서사 정보가 없다.
            (예: "차도윤" → 이름만으로는 성격을 알 수 없음)
            이 경우 캐릭터의 성격은 창작자가 단 태그가 거의 전담하므로,
            임베딩에서 이름의 비중을 낮추고 태그를 핵심 신호로 삼는다.

          이는 수집 불가능한 description 필드를 포기한 데 따른 보완책으로,
          '장르별 정보 구조의 차이'를 임베딩 단계에서 반영한 설계다.
        """
        genre = character.get("genre", "")
        tags_raw = character.get("tags_raw", "").replace(",", " ")
        tags_norm = character.get("tags_normalized", "").replace(",", " ")
        category = character.get("category", "").replace(",", " ")

        if genre == "simulation":
            # 시뮬레이션: 제목이 서사 요약 → 제목을 핵심 신호로.
            return (
                f"{character['name']}. "
                f"{category}. {tags_raw}. {tags_norm}"
            ).strip()
        else:
            # 1:1 롤플레잉: 이름은 서사 정보가 빈약 → 태그를 핵심 신호로.
            # 이름은 'OO 캐릭터'로 보조적으로만 언급한다.
            return (
                f"{character['name']} 캐릭터. "
                f"성격과 분위기: {tags_raw}. {tags_norm}. {category}"
            ).strip()

    @staticmethod
    def _build_metadata(character: dict) -> dict:
        """스키마 v2 전체 필드를 ChromaDB 메타데이터로 가공한다."""
        return {
            "character_id": str(character["character_id"]),
            "name": str(character["name"]),
            "creator_name": str(character.get("creator_name", "")),
            "creator_followers": int(character.get("creator_followers", 0)),
            "genre": str(character.get("genre", "")),
            "category": str(character.get("category", "")),
            "target_orientation": str(character.get("target_orientation", "")),
            "conversation_count": int(character.get("conversation_count", 0)),
            "like_count": int(character.get("like_count", 0)),
            "comment_count": int(character.get("comment_count", 0)),
            "updated_at": str(character.get("updated_at", "")),
            "tags_raw": str(character.get("tags_raw", "")),
            "tags_normalized": str(character.get("tags_normalized", "")),
            # description 도 메타데이터에 보존 → 멘트 생성 등에서 재사용.
            "description": str(character.get("description", "")),
        }

    def add_characters(self, character_list: list[dict]) -> int:
        """캐릭터 리스트를 ChromaDB에 적재. 반환: 적재 수."""
        if not character_list:
            print("⚠️  추가할 캐릭터가 없습니다.")
            return 0

        ids = [str(c["character_id"]) for c in character_list]
        documents = [self._build_embedding_text(c) for c in character_list]
        metadatas = [self._build_metadata(c) for c in character_list]

        print(f"🧮 {len(documents)}개 캐릭터를 임베딩하는 중...")
        embeddings = self.embedder.embed_texts(documents)

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )
        print(f"✅ {len(ids)}개 캐릭터를 ChromaDB에 저장 완료.")
        return len(ids)

    # ------------------------------------------------------------------
    # v3 추가: 크롤링 날것 데이터 → Claude 정규화 → 멱등 적재
    # ------------------------------------------------------------------
    def upsert_crawled_characters(self, raw_characters: list[dict]) -> int:
        """
        크롤러가 수집한 날것 데이터를 정규화하여 ChromaDB에 멱등 적재한다.

        멱등성(Idempotency): 같은 story_id 를 다시 수집해도 중복 행이
        생기지 않는다. ChromaDB의 upsert 가 'id 존재 시 갱신, 없으면 삽입'을
        보장하므로, 크롤러를 매일 돌려도 데이터가 오염되지 않는다.

        반환: 적재(갱신+삽입)된 캐릭터 수.
        """
        if not raw_characters:
            print("⚠️  적재할 크롤링 데이터가 없습니다.")
            return 0

        # 1) 날것 데이터를 v2 스키마 규격으로 정규화.
        normalized = [self._normalize_raw(raw) for raw in raw_characters]

        # 2) 정규화된 데이터를 기존 add_characters 와 동일 방식으로 적재.
        ids = [c["character_id"] for c in normalized]
        documents = [self._build_embedding_text(c) for c in normalized]
        metadatas = [self._build_metadata(c) for c in normalized]

        print(f"🧮 크롤링 캐릭터 {len(documents)}개를 임베딩하는 중...")
        embeddings = self.embedder.embed_texts(documents)

        # upsert: story_id 기반 멱등 적재.
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )
        print(f"✅ 크롤링 캐릭터 {len(ids)}개 멱등 적재 완료.")
        return len(ids)

    def _normalize_raw(self, raw: dict) -> dict:
        """
        크롤러 날것 데이터 1건을 v2 스키마 규격의 캐릭터 dict 로 변환한다.

        - chips(['시뮬레이션','남성향','SF/판타지'])를
          genre / target_orientation / category 로 분해.
        - updated_at_raw('2026.05.10') → 'YYYY-MM-DD'.
        - tags_normalized 는 Claude 로 29개 화이트리스트에 매핑.
        """
        chips = raw.get("chips", [])

        # --- chip 분해: 미리 정의된 매핑 테이블로 분류 ---
        genre_map = {"1:1 롤플레잉": "roleplay_1on1", "시뮬레이션": "simulation"}
        orient_map = {"남성향": "male", "여성향": "female", "전체": "unspecified"}
        category_map = {
            "로맨스": "romance", "BL": "bl", "GL": "gl",
            "SF/판타지": "sf_fantasy", "로판": "romance_fantasy",
            "무협": "martial_arts", "일상/현대": "daily_modern",
            "시대": "period", "성인": "adult",
        }

        genre = ""
        orientation = "unspecified"  # 칩에 성향이 없으면 공용으로 간주.
        categories = []
        for chip in chips:
            if chip in genre_map:
                genre = genre_map[chip]
            elif chip in orient_map:
                orientation = orient_map[chip]
            elif chip in category_map:
                categories.append(category_map[chip])
        category_str = ",".join(categories) if categories else "etc"

        # --- 날짜 정규화: '2026.05.10' → '2026-05-10' ---
        updated_at = raw.get("updated_at_raw", "").replace(".", "-").strip()

        # --- tags_raw: '#던전 #생존' → '던전,생존' ---
        tags_raw = (
            raw.get("tags_raw", "")
            .replace("#", "")
            .strip()
        )
        tags_raw = ",".join(t for t in tags_raw.split() if t)

        # --- tags_normalized: Claude 로 29개 화이트리스트에 매핑 ---
        tags_normalized = self._normalize_tags_via_llm(tags_raw)

        return {
            "character_id": raw["story_id"],
            "name": raw["name"],
            "creator_name": raw.get("creator_name", ""),
            "creator_followers": 0,  # 크롤러가 수집 안 함 (추천에 미사용)
            "genre": genre,
            "category": category_str,
            "target_orientation": orientation,
            "conversation_count": raw.get("conversation_count", 0),
            "like_count": raw.get("like_count", 0),
            "comment_count": raw.get("comment_count", 0),
            "updated_at": updated_at,
            "tags_raw": tags_raw,
            "tags_normalized": tags_normalized,
        }

    def _normalize_tags_via_llm(self, tags_raw: str) -> str:
        """
        날것 태그 문자열을 29개 표준 성격 태그 화이트리스트에 매핑한다.

        제작자가 자유 입력한 태그(#무자각안데레 등)는 무한하므로,
        Claude 에게 '가장 가까운 표준 태그로 변환'을 맡긴다.
        genre/category/orientation/제작자명/IP명과 겹치는 태그는 제외한다.
        """
        from anthropic import Anthropic

        valid_tags = [
            "위로", "힐링", "잔잔함", "공감", "설렘", "달달함", "애틋함",
            "다정함", "까칠함", "과묵함", "능글맞음", "티격태격", "코미디",
            "유쾌함", "집착", "얀데레", "긴장감", "어두움", "애증", "생존",
            "성장", "지적임", "전략", "헌신", "청춘", "액션", "판타지",
            "로맨스", "일상",
        ]
        tag_list_str = ", ".join(valid_tags)

        llm = Anthropic()
        system_prompt = (
            "너는 캐릭터 태그를 표준 어휘로 정규화하는 도구다. 규칙:\n"
            f"1. 출력 태그는 반드시 다음 목록 안의 것만 사용한다: "
            f"[{tag_list_str}].\n"
            "2. 입력 태그 중 의미가 가까운 것을 목록 내 태그로 매핑한다.\n"
            "3. 장르·카테고리·제작자명·작품 고유명사에 해당하는 태그는 제외한다.\n"
            "4. 출력은 쉼표로 구분된 태그 문자열 하나. 설명 금지."
        )
        message = llm.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            system=system_prompt,
            messages=[
                {"role": "user", "content": f"입력 태그: {tags_raw}"}
            ],
        )
        result = message.content[0].text.strip()
        # 결과를 화이트리스트로 한 번 더 교차 검증.
        cleaned = [
            t.strip() for t in result.split(",")
            if t.strip() in valid_tags
        ]
        return ",".join(cleaned)
    # ------------------------------------------------------------------
    # ★ v2 개조: 절대 유사도 임계값 가드 추가
    # ------------------------------------------------------------------
    def search_characters(
        self,
        query_text: str,
        target_orientation: str,
        genre: str | None = None,
        top_k: int = 15,
        threshold_guard: float = 0.25,
        min_survivors: int = 3,
    ) -> list[dict]:
        """
        하드 필터를 걸어 유사 캐릭터 후보군을 추출하되,
        raw 유사도가 threshold_guard 미만인 노이즈 후보를 탈락시킨다.

        threshold_guard: 절대 유사도 하한선. raw_similarity(=1-distance)가
            이 값 미만이면 '의미적으로 무관한 노이즈'로 보고 Hard Drop.
            (v1 결함 A: 노이즈가 민맥스로 만점 왜곡되는 문제의 입력단 차단)
        min_survivors: ★ 안전망. 임계값 통과 후보가 이 수보다 적으면,
            가드가 너무 공격적이라 판단하고 '임계값 미적용 결과'로 폴백한다.
            (한국어 임베딩 특성상 raw 유사도가 전반적으로 낮을 수 있어,
             가드만 믿으면 후보가 전멸할 위험이 있다.)

        반환: [{"id","document","metadata","distance","raw_similarity"}, ...]
        """
        query_vector = self.embedder.embed_query(query_text)

        # where 필터: 고정 성향(+공용) + 선택적 장르.
        conditions = [
            {"target_orientation": {"$in": [target_orientation, "unspecified"]}}
        ]
        if genre:
            conditions.append({"genre": {"$eq": genre}})
        where_filter = conditions[0] if len(conditions) == 1 else {"$and": conditions}

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        # 전체 후보를 dict 로 재조립하면서 raw_similarity 를 계산해 둔다.
        all_hits = []
        for i in range(len(ids)):
            raw_sim = max(0.0, 1.0 - dists[i])
            all_hits.append(
                {
                    "id": ids[i],
                    "document": docs[i],
                    "metadata": metas[i],
                    "distance": dists[i],
                    "raw_similarity": raw_sim,
                }
            )

        # --- 임계값 가드 적용 ---
        survivors = [h for h in all_hits if h["raw_similarity"] >= threshold_guard]

        # --- 안전망: 통과 후보가 너무 적으면 가드 미적용으로 폴백 ---
        if len(survivors) < min_survivors:
            print(
                f"⚠️  임계값({threshold_guard}) 통과 후보가 "
                f"{len(survivors)}개뿐 → 가드를 완화해 전체 후보를 사용합니다."
            )
            return all_hits

        return survivors

    def count(self) -> int:
        return self.collection.count()