"""src/embedding/embedder.py

역할: 텍스트(자연어 문장)를 고차원 벡터(숫자 좌표 리스트)로 변환한다.
     OpenAI의 text-embedding-3-small 모델을 raw SDK로 직접 호출한다.

이 클래스는 '번역기'다. 사람의 언어(문장)를 컴퓨터가 거리(유사도)를
계산할 수 있는 숫자 공간의 좌표로 옮긴다. ChromaDB는 이 좌표만 다룬다.
"""
from openai import OpenAI


class OpenAIEmbedder:
    """OpenAI 임베딩 API를 감싸는 얇은 래퍼 클래스."""

    # text-embedding-3-small이 출력하는 벡터의 차원 수.
    # vector_store가 컬렉션을 만들 때 차원을 알아야 하므로 상수로 노출해 둔다.
    DIMENSION = 1536

    def __init__(self, model: str = "text-embedding-3-small"):
        """
        OpenAI() 는 생성자에 키를 직접 넣지 않아도,
        환경변수 OPENAI_API_KEY 를 자동으로 찾아 사용한다.
        (.env 로딩은 프로그램 진입점에서 load_dotenv() 로 처리한다.)
        """
        self.client = OpenAI()
        self.model = model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        여러 개의 문장을 한 번의 API 호출로 묶어 벡터 리스트로 변환한다.

        문장 100개를 100번 호출하지 않고 1번에 보내는 것을 '배치(batch)'라 한다.
        호출 횟수가 줄어 속도도 빠르고 비용/요청 제한에도 유리하다.

        반환: [[0.01, -0.23, ...], [...], ...]  (입력 순서와 1:1로 대응)
        """
        if not texts:
            return []

        # OpenAI 임베딩 엔드포인트 호출.
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,  # 문자열 리스트를 그대로 넘기면 배치 처리된다.
        )

        # response.data 는 입력 순서가 보장된 결과 객체들의 리스트다.
        # 각 객체의 .embedding 필드가 실제 숫자 벡터다.
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        """
        문장 '하나'(주로 유저의 검색 질문)를 벡터 하나로 변환하는 편의 메서드.
        내부적으로는 embed_texts 를 재사용하고 첫 번째 결과만 꺼낸다.
        """
        return self.embed_texts([text])[0]