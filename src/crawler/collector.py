"""src/crawler/collector.py  (v3 신규)

역할: 동적 렌더링 웹 페이지에서 캐릭터 데이터를 수집하는 크롤러.

설계 원칙:
  - Playwright 기반 '실전 구조'로 작성하되, 시연 타깃은 로컬 가상 HTML.
    → 실서비스 전환 시 TARGET_URL 과 CSS 셀렉터만 교체하면 된다.
  - Rate Limit 가드(무작위 지연 + User-Agent)를 장착한다.

★ 법적 주의 (코드 사용자 필독):
  커스텀 User-Agent 와 지연은 '예의 바른 크롤링'을 위한 것일 뿐,
  그 자체가 수집 허가를 의미하지 않는다. 실제 웹사이트 수집의
  적법성은 오직 해당 사이트의 이용약관(ToS)과 robots.txt 가 정한다.
  본 프로젝트는 기업 연계 협의 하에 공개 데이터 수집 범위를 승인받았다.
"""
import random
import time
from pathlib import Path

from playwright.sync_api import sync_playwright


class DynamicCharacterCrawler:
    """Playwright 기반 동적 웹 캐릭터 크롤러."""

    # 학술 연구 목적을 명시하는 커스텀 User-Agent.
    # (정중한 신원 표시일 뿐, 수집 허가의 근거는 아님)
    USER_AGENT = (
        "WrtnCapstoneResearchBot/1.0 "
        "(Academic capstone project; contact: student@university.ac.kr)"
    )

    def __init__(self, target_url: str | None = None):
        """
        target_url: 수집 대상 페이지 URL.
            None 이면 로컬 가상 타깃(data/mock_site/characters.html)을 쓴다.
            실서비스 전환 시 실제 URL을 주입하면 된다.
        """
        if target_url:
            self.target_url = target_url
        else:
            # 로컬 가상 HTML 파일을 file:// URL 로 변환.
            root = Path(__file__).resolve().parents[2]
            html_path = root / "data" / "mock_site" / "characters.html"
            self.target_url = html_path.as_uri()  # file:///... 형식

    # ------------------------------------------------------------------
    # Rate Limit 가드: 요청 간 무작위 지연
    # ------------------------------------------------------------------
    @staticmethod
    def _polite_delay() -> None:
        """
        2~5초 사이 무작위 지연(Random Jitter)을 준다.
        일정한 간격의 기계적 요청은 서버에 부담을 주고 차단 대상이 되므로,
        사람처럼 불규칙한 간격으로 요청해 상대 서버를 배려한다.
        """
        delay = random.uniform(2.0, 5.0)
        print(f"  ⏳ Rate limit 가드: {delay:.1f}초 대기...")
        time.sleep(delay)

    # ------------------------------------------------------------------
    # 수집: 페이지를 열어 캐릭터 카드들을 파싱
    # ------------------------------------------------------------------
    def crawl(self) -> list[dict]:
        """
        타깃 페이지를 브라우저로 렌더링하고, 캐릭터 카드를 파싱해
        딕셔너리 리스트로 반환한다.

        반환 원소(날것 데이터, 아직 정규화 전):
          {"story_id","name","creator_name","chips":[...],
           "tags_raw":"...", "conversation_count":int,
           "like_count":int, "comment_count":int, "updated_at_raw":"..."}
        """
        print(f"🕷️  크롤러 가동 — 타깃: {self.target_url}")
        collected = []

        with sync_playwright() as p:
            # headless=True: 화면 없이 백그라운드로 브라우저 구동.
            browser = p.chromium.launch(headless=True)
            # 커스텀 User-Agent 를 컨텍스트에 설정.
            context = browser.new_context(user_agent=self.USER_AGENT)
            page = context.new_page()

            # 페이지 로드. JS 렌더링 페이지라면 이 시점에 React 등이 그려진다.
            page.goto(self.target_url)
            # 동적 콘텐츠가 다 그려질 때까지 대기 (실전 페이지 대비).
            page.wait_for_selector(".story-card")

            # 모든 캐릭터 카드 요소를 수집.
            cards = page.query_selector_all(".story-card")
            print(f"  📋 카드 {len(cards)}개 발견")

            for card in cards:
                # 매 카드 파싱 사이에 예의 지연 주입.
                self._polite_delay()

                # --- CSS 셀렉터로 각 필드 추출 ---
                # 실서비스 전환 시 이 셀렉터 문자열들만 교체하면 된다.
                story_id = card.get_attribute("data-story-id")
                name = card.query_selector(".story-title").inner_text()
                creator = card.query_selector(".creator-name").inner_text()

                chip_els = card.query_selector_all(".chip")
                chips = [c.inner_text().strip() for c in chip_els]

                tags_raw = card.query_selector(".tags").inner_text()
                conv = card.query_selector(".metric-conversation").inner_text()
                like = card.query_selector(".metric-like").inner_text()
                comment = card.query_selector(".metric-comment").inner_text()
                updated = card.query_selector(".updated-at").inner_text()

                collected.append(
                    {
                        "story_id": story_id,
                        "name": name.strip(),
                        "creator_name": creator.strip(),
                        "chips": chips,
                        # 태그 문자열에서 '#'와 공백 정리.
                        "tags_raw": tags_raw.strip(),
                        # 메트릭은 문자열 → 정수 변환.
                        "conversation_count": self._to_int(conv),
                        "like_count": self._to_int(like),
                        "comment_count": self._to_int(comment),
                        "updated_at_raw": updated.strip(),
                    }
                )
                print(f"  ✅ 수집: {name.strip()}")

            browser.close()

        print(f"🕷️  크롤링 완료 — 총 {len(collected)}개 수집\n")
        return collected

    @staticmethod
    def _to_int(text: str) -> int:
        """'182000' 같은 숫자 문자열을 정수로. 변환 실패 시 0."""
        cleaned = text.strip().replace(",", "")
        try:
            return int(cleaned)
        except ValueError:
            return 0