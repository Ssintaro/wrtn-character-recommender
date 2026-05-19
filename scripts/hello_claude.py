"""scripts/hello_claude.py
오늘의 목표: 가상 캐릭터 JSON을 읽어 Claude API로 추천을 받아본다.
아직 RAG/임베딩은 없다. '데이터 -> API -> 응답' 파이프라인만 검증한다.
"""
import json
from pathlib import Path
from anthropic import Anthropic
from dotenv import load_dotenv

# 1) .env 파일의 API 키를 환경변수로 로드
load_dotenv()

# 2) Anthropic 클라이언트 생성 (환경변수에서 키를 자동으로 읽음)
client = Anthropic()

# 3) 4번 단계에서 만든 가상 캐릭터 데이터(JSON) 로드
# (Path를 사용해 Windows/Linux 어디서든 경로가 안 깨지게 자동 계산)
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "mock" / "characters.json"

try:
    with open(DATA_PATH, encoding="utf-8") as f:
        characters = json.load(f)
    print(f"✅ 가상 캐릭터 {len(characters)}명 로드 완료!")
except FileNotFoundError:
    print(f"❌ 에러: {DATA_PATH} 위치에 파일이 없습니다. 폴더 위치를 다시 확인해 주세요.")
    exit(1)

# 4) 캐릭터 목록을 프롬프트(글자) 덩어리로 변환
character_block = "\n".join(
    f"- [{c['character_id']}] {c['name']} | {c['one_liner']} "
    f"| 태그: {', '.join(c['tags'])} | 소개: {c['description']}"
    for c in characters
)

# 5) 사용자의 현재 가상 상황 (테스트용 입력값)
user_context = "요즘 캡스톤 디자인 때문에 스트레스를 너무 많이 받아서 지쳤어. 나한테 따뜻한 위로와 공감을 해줄 친구가 필요해."

# 6) Claude에게 보낼 프롬프트 조립 (XML 태그 활용)
prompt = f"""너는 AI 캐릭터 추천 큐레이터다.
아래 <캐릭터 목록> 중 사용자의 현재 상황에 가장 잘 맞는 캐릭터 딱 1명을 골라,
추천하는 이유를 2~3문장으로 유저에게 친근하게 설명하라.

<캐릭터 목록>
{character_block}

<사용자의 현재 상황>
{user_context}
"""

print("🚀 Claude API 호출 중... 잠시만 기다려주세요.")

# 7) Claude API 호출 (가성비가 좋은 Haiku 모델 사용)
message = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=500,
    messages=[{"role": "user", "content": prompt}],
)

# 8) 최종 추천 결과 출력
print("\n" + "="*20 + " Claude의 추천 결과 " + "="*20)
print(message.content[0].text)
print("="*58)