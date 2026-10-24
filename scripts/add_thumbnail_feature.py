"""Original DOM/metadata thumbnail algorithm; portable paths and overwrite guards."""
from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections import Counter, defaultdict
from html import unescape
from pathlib import Path
from urllib.parse import (
    parse_qsl,
    unquote,
    urlencode,
    urljoin,
    urlsplit,
    urlunsplit,
)

# Browser dependency is optional for the training pipeline and CLI --help.


# ============================================================
# 썸네일 제목으로 보기 어려운 UI / 기능 문구
# ============================================================

CTA_TEXTS = {
    "go",
    "more",
    "더보기",
    "전체보기",
    "상세보기",
    "상세안내",
    "자세히보기",
    "다운로드",
    "이전",
    "다음",
    "바로가기",
    "열기",
    "닫기",
    "홈",
    "home",
    "재생",
    "정지",
    "시작",
    "인쇄하기",
    "주소복사",
    "맨위로",
    "top",
    "새창",
    "새창열기",
    "새 창 열림",
    "(새 창 열림)",
}

GENERIC_LINK_TEXTS = {
    "로그인",
    "로그아웃",
    "회원가입",
    "검색",
    "메뉴",
    "사이트맵",
    "페이스북",
    "인스타그램",
    "유튜브",
    "블로그",
    "트위터",
    "x",
    "카카오톡",
}

# alt 자체가 아래처럼 명시적으로 UI/아이콘임을 말하면
# thumbnail card가 아니라 icon/function link로 취급한다.
ALT_ICON_HINTS = (
    "아이콘",
    "icon",
    "로고",
    "logo",
    "웹 접근성",
    "wa인증",
    "인증마크",
)

# URL/파일명에 아래 패턴이 있으면 "아이콘일 가능성"이 높다.
# 단독으로 False 처리하지 않고, 다른 근거와 함께 사용한다.
URL_ICON_HINT_RE = re.compile(
    r"(?:^|[/_.\-])("
    r"ico(?:n)?|logo|sns|social|arrow|btn|button|"
    r"external[-_]?link|link[-_]?blank|weather|wea|"
    r"facebook|instagram|youtube|twitter|kakao"
    r")(?:[/_.\-]|$)",
    re.IGNORECASE,
)

FILETYPE_ICON_BASENAME_RE = re.compile(
    r"^(?:xls[x]?|pdf|doc[x]?|hwp|zip|ppt[x]?|file)"
    r"(?:[_\-].*)?\.(?:png|gif|svg|jpe?g|webp)$",
    re.IGNORECASE,
)

IMAGE_EXT_RE = re.compile(
    r"\.(?:jpe?g|png|gif|webp|bmp|svg|avif|tiff?|ico)$",
    re.IGNORECASE,
)

WEB_ROUTE_EXT_RE = re.compile(
    r"\.(?:do|jsp|php|aspx?|html?|cgi)(?:[?#]|$)",
    re.IGNORECASE,
)


# ============================================================
# Utility
# ============================================================

def normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_for_label(text: str | None) -> str:
    text = normalize_text(text).lower()
    text = re.sub(r"[>\-–—»›→]+$", "", text).strip()
    return text


def is_generic_link_text(text: str | None) -> bool:
    """
    '더보기', '다운로드', '재생' 등
    제목이라기보다 기능/네비게이션을 표현하는 문구인지 검사.
    """
    text = normalize_for_label(text)

    if not text:
        return True

    if text in CTA_TEXTS:
        return True

    if text in GENERIC_LINK_TEXTS:
        return True

    # 약간의 표기 변형도 제거
    compact = re.sub(r"[\s_\-]+", "", text)

    if compact in {
        "새창열림",
        "새창아이콘",
        "인쇄하기",
        "주소복사",
        "맨위로",
        "자세히보기",
        "상세보기",
        "전체보기",
        "다운로드",
    }:
        return True

    return False


def has_meaningful_text(text: str | None) -> bool:
    """
    같은 링크 안에 실제 제목/설명으로 볼 만한 텍스트가 있는지.
    """
    text = normalize_text(text)

    if not text:
        return False

    compact = re.sub(r"\s+", "", text)

    if len(compact) < 2:
        return False

    if not re.search(r"[가-힣A-Za-z0-9]", text):
        return False

    if is_generic_link_text(text):
        return False

    return True


def strip_visual_suffix(text: str | None) -> str:
    """
    가이드라인의 그래픽 썸네일 비교를 위해 alt 끝의
    '이미지', '아이콘', '로고' 같은 시각 유형 접미를 제거한다.

    예:
      '교통과 이미지' -> '교통과'
      '계약정보공개 아이콘 이미지' -> '계약정보공개'
      'KISTI 로고' -> 'kisti'
    """
    text = normalize_for_label(text)

    if not text:
        return ""

    # 여러 접미가 연달아 붙은 경우도 반복 제거한다.
    suffix_re = re.compile(
        r"(?:\s*(?:이미지|아이콘|로고|image|icon|logo|사진))+$",
        re.IGNORECASE,
    )
    text = suffix_re.sub("", text).strip()

    # 비교 안정화를 위해 끝의 구두점/괄호 잔여물을 정리한다.
    text = re.sub(r"[\s:：,，.。·|/\\()\[\]{}<>_-]+$", "", text).strip()
    return text


def normalize_for_overlap(text: str | None) -> str:
    """
    alt와 같은-link 제목 텍스트의 포함 관계 확인용 정규화.
    띄어쓰기와 가벼운 구두점 차이는 무시한다.
    """
    text = normalize_for_label(text)
    return re.sub(r"[^0-9a-z가-힣]+", "", text, flags=re.IGNORECASE)


def alt_repeats_link_title(
    alt_text: str | None,
    link_text: str | None,
) -> bool:
    """
    작은 아이콘/그래픽이라도 alt의 핵심 내용이 같은 링크의
    제목 텍스트에 그대로 포함되면 '이미지 + 제목 텍스트가
    같은 링크에 묶인 그래픽 썸네일'로 인정하기 위한 규칙.

    가이드라인의 'alt 끝의 이미지/아이콘 접미를 떼고 비교'를 반영한다.
    """
    alt_core = strip_visual_suffix(alt_text)
    link_text = normalize_text(link_text)

    if not alt_core or not link_text:
        return False

    if is_generic_link_text(alt_core):
        return False

    alt_norm = normalize_for_overlap(alt_core)
    link_norm = normalize_for_overlap(link_text)

    # 한 글자 같은 지나치게 짧은 우연 매칭 방지
    if len(alt_norm) < 2 or len(link_norm) < 2:
        return False

    # 가이드라인상 핵심은 '링크 텍스트가 alt 핵심 내용을 포함하는가'.
    return alt_norm in link_norm


def looks_like_url_or_path(value: str | None) -> bool:
    """
    link_dest가 '표시 텍스트'가 아니라 URL/상대경로/anchor/query에
    가까운 값인지 판별한다.

    예:
      https://...
      /portal/main.do
      #none
      ?page=2
      category_list.do?ids1=A...
      viewPublicationList.do?bg_no=494
    """
    value = normalize_text(value)

    if not value:
        return True

    low = value.lower()

    if low.startswith(
        (
            "http://",
            "https://",
            "//",
            "/",
            "./",
            "../",
            "#",
            "?",
            "javascript:",
            "mailto:",
            "tel:",
        )
    ):
        return True

    # 공백 없이 route-like 파일 경로인 경우
    if " " not in value and WEB_ROUTE_EXT_RE.search(value):
        return True

    # 공백 없이 path separator가 있는 경우
    if " " not in value and "/" in value:
        return True

    # key=value 형태가 포함된 query/route
    if " " not in value and re.search(r"(?:^|[?&])[A-Za-z0-9_.~-]+=", value):
        return True

    return False


def canonical_url(raw_url: str | None, base_url: str) -> str | None:
    """
    URL 비교용 canonical form.

    - 상대 URL → 절대 URL
    - fragment 제거
    - query parameter 순서 정렬
    - hostname 소문자화
    """
    if not raw_url:
        return None

    raw_url = unescape(str(raw_url).strip())

    if not raw_url:
        return None

    try:
        absolute = urljoin(base_url, raw_url)
        parsed = urlsplit(absolute)

        host = (parsed.hostname or "").lower()

        if parsed.port:
            netloc = f"{host}:{parsed.port}"
        else:
            netloc = host

        query = parse_qsl(parsed.query, keep_blank_values=True)
        query.sort()

        return urlunsplit(
            (
                parsed.scheme.lower(),
                netloc,
                unquote(parsed.path),
                urlencode(query, doseq=True),
                "",
            )
        )

    except Exception:
        return None


def image_basename(url: str | None) -> str:
    if not url:
        return ""

    try:
        return Path(unquote(urlsplit(url).path)).name.lower()
    except Exception:
        return ""


def target_has_real_image_extension(url: str | None) -> bool:
    name = image_basename(url)
    return bool(IMAGE_EXT_RE.search(name))


def alt_explicitly_says_icon_or_logo(alt_text: str | None) -> bool:
    """
    '개인정보처리 아이콘 이미지', '새창 아이콘', '기관 로고' 등.
    공식 가이드에서 icon은 thumbnail과 별도 역할로 취급하므로
    이런 경우는 thumbnail=False로 보수적으로 처리한다.
    """
    text = normalize_for_label(alt_text)

    if not text:
        return False

    return any(hint in text for hint in ALT_ICON_HINTS)


def url_has_icon_hint(image_url: str | None) -> bool:
    if not image_url:
        return False

    try:
        parsed = urlsplit(image_url)
        target = unquote(parsed.path).lower()
        basename = Path(target).name
    except Exception:
        target = str(image_url).lower()
        basename = Path(target).name

    # /thumbnail/archive/pdf/... 같은 "콘텐츠 디렉터리명"은 제외하고,
    # 실제 파일명 자체가 pdf.png / xls.png처럼 파일형식 아이콘일 때만 잡는다.
    if FILETYPE_ICON_BASENAME_RE.fullmatch(basename):
        return True

    return bool(URL_ICON_HINT_RE.search(target))


def is_small_ui_image(dom: dict | None) -> bool:
    """
    렌더링 크기와 natural size가 모두 작을 때만 작은 UI 아이콘으로 본다.

    CSS 때문에 한쪽 값만 작아지는 경우를 피하기 위해
    rendered / natural 중 확인 가능한 값들을 함께 본다.
    """
    if not dom:
        return False

    width = dom.get("rendered_width")
    height = dom.get("rendered_height")
    natural_width = dom.get("natural_width")
    natural_height = dom.get("natural_height")

    rendered_small = (
        isinstance(width, (int, float))
        and isinstance(height, (int, float))
        and width > 0
        and height > 0
        and width <= 48
        and height <= 48
    )

    natural_small = (
        isinstance(natural_width, (int, float))
        and isinstance(natural_height, (int, float))
        and natural_width > 0
        and natural_height > 0
        and natural_width <= 64
        and natural_height <= 64
    )

    return rendered_small or natural_small


def obvious_non_thumbnail_visual(record: dict, dom: dict | None = None) -> tuple[bool, str | None]:
    """
    링크 안 이미지이기는 하지만 thumbnail card가 아니라
    icon/logo/file-format/control image일 가능성이 매우 높은 경우를 거른다.

    너무 공격적으로 제외하지 않기 위해:
      - alt가 명시적으로 icon/logo라고 말하는 경우 -> 바로 제외
      - 아주 작은 이미지 -> 제외
      - URL이 icon-like + alt/DOM 크기 근거가 있을 때 -> 제외
    """
    alt_text = record.get("alt_text", "")
    image_url = record.get("image_url", "")

    if alt_explicitly_says_icon_or_logo(alt_text):
        return True, "explicit_icon_or_logo_alt"

    if is_small_ui_image(dom):
        return True, "small_ui_image"

    if url_has_icon_hint(image_url):
        # icon/logo/sns/arrow/btn처럼 파일 경로 자체가
        # UI 이미지임을 강하게 나타내는 경우는 thumbnail card에서 제외한다.
        return True, "icon_like_image_url"

    return False, None


# ============================================================
# metadata fallback
# ============================================================

def fallback_thumbnail_from_metadata(record: dict) -> tuple[bool, str]:
    """
    DOM에서 image_url에 해당하는 <img>를 못 찾은 경우에만 사용.

    데이터셋의 in_link=True라는 사실은 이미 알고 있으므로,
    link_dest가 실제 '제목/설명 텍스트'처럼 보이는지를 이용한다.

    최종 규칙:
      - alt의 핵심 내용(끝의 '이미지/아이콘/로고' 제거)이
        link_dest 제목에 반복되면 작은 아이콘/그래픽이어도 True.
      - 그렇지 않은 icon/logo/UI는 기존처럼 False.
    """
    link_dest = normalize_text(record.get("link_dest"))

    if not link_dest:
        return False, "fallback_empty_link_dest"

    if looks_like_url_or_path(link_dest):
        return False, "fallback_url_or_path_link_dest"

    if is_generic_link_text(link_dest):
        return False, "fallback_generic_ui_text"

    if not has_meaningful_text(link_dest):
        return False, "fallback_non_meaningful_link_text"

    # --------------------------------------------------------
    # 작은 그래픽/아이콘 예외
    # alt 핵심이 같은 링크 제목에 반복되면 graphic thumbnail로 인정.
    # obvious_non_thumbnail_visual보다 반드시 먼저 검사한다.
    # --------------------------------------------------------
    if alt_repeats_link_title(
        record.get("alt_text"),
        link_dest,
    ):
        return True, "fallback_alt_repeats_link_title"

    non_thumbnail, reason = obvious_non_thumbnail_visual(record, dom=None)

    if non_thumbnail:
        return False, f"fallback_{reason}"

    return True, "fallback_text_link_dest"


# ============================================================
# JSONL
# ============================================================

def load_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"JSON decode error at physical line {line_number}: {e}"
                ) from e

    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ============================================================
# Browser / DOM
# ============================================================

async def collect_images(page) -> list[dict]:
    """
    현재 DOM의 모든 <img>를 수집한다.

    lazy loading 대응을 위해 src뿐 아니라 data-src,
    data-original, srcset 등을 가져온다.

    최종 버전:
    같은 image_url이 여러 카드에서 재사용되는 경우를 구분할 수 있도록
    각 이미지가 속한 <a>의 visible text와 href도 함께 저장한다.
    """

    return await page.locator("img").evaluate_all(
        """
        imgs => imgs.map((img, index) => {
            const scanId = String(index);
            img.setAttribute("data-koaltq-scan-id", scanId);

            function srcsetUrls(value) {
                if (!value) return [];

                return value
                    .split(",")
                    .map(x => x.trim())
                    .filter(Boolean)
                    .map(x => x.split(/\\s+/)[0]);
            }

            const urls = [
                img.currentSrc || "",
                img.src || "",
                img.getAttribute("src") || "",
                img.getAttribute("data-src") || "",
                img.getAttribute("data-original") || "",
                img.getAttribute("data-original-src") || "",
                img.getAttribute("data-lazy-src") || "",
                img.getAttribute("data-url") || "",
                ...srcsetUrls(img.getAttribute("srcset")),
                ...srcsetUrls(img.getAttribute("data-srcset"))
            ];

            const anchor = img.closest("a[href]");
            const anchorText = anchor
                ? (anchor.innerText || "").replace(/\\s+/g, " ").trim()
                : "";
            const anchorHref = anchor ? (anchor.href || "") : "";

            return {
                scan_id: scanId,
                alt: img.getAttribute("alt") || "",
                urls: [...new Set(urls.filter(Boolean))],
                anchor_text: anchorText,
                anchor_href: anchorHref
            };
        })
        """
    )




async def auto_scroll(page) -> None:
    """
    이미지가 DOM에 아직 등장하지 않은 경우
    lazy loading을 유도하기 위해 한 번만 수행.
    """

    try:
        await page.evaluate(
            """
            async () => {
                const sleep = ms =>
                    new Promise(resolve => setTimeout(resolve, ms));

                const step = Math.max(
                    500,
                    Math.floor(window.innerHeight * 0.8)
                );

                let previous = -1;

                for (let i = 0; i < 40; i++) {
                    const current = window.scrollY;

                    if (current === previous)
                        break;

                    previous = current;

                    window.scrollBy(0, step);
                    await sleep(120);

                    if (
                        window.innerHeight + window.scrollY
                        >= document.documentElement.scrollHeight - 10
                    ) {
                        break;
                    }
                }

                await sleep(500);
                window.scrollTo(0, 0);
            }
            """
        )
    except Exception:
        pass


def _normalized_text_equal(a: str | None, b: str | None) -> bool:
    """
    DOM candidate 매칭에서만 사용하는 보수적인 텍스트 동등 비교.
    공백/가벼운 구두점 차이만 무시하고, 부분 포함만으로는 같다고 보지 않는다.
    """
    a_norm = normalize_for_overlap(a)
    b_norm = normalize_for_overlap(b)
    return bool(a_norm and b_norm and a_norm == b_norm)


def _select_candidate(
    candidates: list[dict],
    record: dict,
    page_url: str,
    method_prefix: str,
) -> tuple[dict | None, str]:
    """
    동일 URL/basename 후보가 여러 개인 경우 올바른 DOM <img>를 고른다.

    우선순위:
      1) dataset alt_text와 DOM img alt의 exact match
      2) dataset link_dest가 표시 텍스트라면 같은 <a>의 text와 exact match
      3) dataset link_dest가 URL/path라면 같은 <a>의 href와 canonical match

    끝까지 하나로 좁혀지지 않으면 임의로 첫 번째 이미지를 고르지 않는다.
    그 경우 None을 반환하여 상위 로직의 metadata fallback을 사용한다.
    """
    if not candidates:
        return None, f"{method_prefix}_no_candidate"

    if len(candidates) == 1:
        return candidates[0], method_prefix

    target_alt = normalize_text(record.get("alt_text"))

    # 1) alt exact match
    if target_alt:
        alt_matches = [
            image
            for image in candidates
            if normalize_text(image.get("alt")) == target_alt
        ]

        if len(alt_matches) == 1:
            return alt_matches[0], f"{method_prefix}_and_alt"

        # alt가 여러 후보에 공통으로 반복되면 다음 근거를 계속 본다.
        if len(alt_matches) > 1:
            candidates = alt_matches

    link_dest = normalize_text(record.get("link_dest"))

    if link_dest:
        # 2) link_dest가 href/경로형인 경우
        if looks_like_url_or_path(link_dest):
            target_href = canonical_url(link_dest, page_url)

            if target_href:
                href_matches = [
                    image
                    for image in candidates
                    if canonical_url(image.get("anchor_href"), page_url) == target_href
                ]

                if len(href_matches) == 1:
                    return href_matches[0], f"{method_prefix}_and_href"

                if len(href_matches) > 1:
                    candidates = href_matches

        # 3) link_dest가 표시 텍스트인 경우
        else:
            text_matches = [
                image
                for image in candidates
                if _normalized_text_equal(
                    image.get("anchor_text"),
                    link_dest,
                )
            ]

            if len(text_matches) == 1:
                return text_matches[0], f"{method_prefix}_and_link_text"

            if len(text_matches) > 1:
                candidates = text_matches

    if len(candidates) == 1:
        return candidates[0], f"{method_prefix}_resolved"

    # 중요:
    # 동일 아이콘/로고가 여러 카드에서 재사용되는 상황에서
    # first candidate를 임의 선택하면 다른 <a>의 구조를 검사하게 된다.
    # 확실한 근거가 없으면 ambiguous로 두고 metadata fallback으로 넘긴다.
    return None, f"{method_prefix}_ambiguous"


def match_image(
    images: list[dict],
    record: dict,
    page_url: str,
) -> tuple[dict | None, str]:
    """
    record의 image_url과 실제 DOM <img>를 매칭한다.

    우선순위:
      1. canonical URL match
      2. 실제 이미지 파일명 basename match
      3. DOM alt == dataset alt가 페이지에서 유일한 경우

    같은 URL/basename이 여러 DOM 요소에서 재사용되면
    alt + link_dest(anchor text 또는 href)를 이용해 후보를 구분한다.
    그래도 모호하면 첫 번째 후보를 임의 선택하지 않는다.
    """

    target_url = canonical_url(
        record.get("image_url"),
        page_url,
    )

    target_alt = normalize_text(record.get("alt_text"))

    # --------------------------------------------------------
    # 1. URL exact match
    # --------------------------------------------------------

    exact_matches = []

    for image in images:
        normalized_urls = {
            canonical_url(url, page_url)
            for url in image["urls"]
        }

        normalized_urls.discard(None)

        if target_url and target_url in normalized_urls:
            exact_matches.append(image)

    if exact_matches:
        matched, method = _select_candidate(
            exact_matches,
            record,
            page_url,
            "exact_url",
        )

        if matched is not None:
            return matched, method

        # exact URL 후보가 있었지만 여러 개라 모호한 경우에는
        # basename/alt로 다른 이미지를 억지로 찾지 않고 fallback으로 넘긴다.
        if method == "exact_url_ambiguous":
            return None, method

    # --------------------------------------------------------
    # 2. basename fallback
    #
    # abc.jpg 같은 실제 이미지 파일에서만 허용.
    # view.do, image.php처럼 basename이 의미 없는 URL은 사용하지 않는다.
    # --------------------------------------------------------

    if target_has_real_image_extension(record.get("image_url")):
        target_name = image_basename(record.get("image_url"))

        basename_matches = []

        for image in images:
            matched_basename = False

            for raw_url in image["urls"]:
                candidate = canonical_url(raw_url, page_url)

                if (
                    candidate
                    and image_basename(candidate) == target_name
                ):
                    matched_basename = True
                    break

            if matched_basename:
                basename_matches.append(image)

        if basename_matches:
            matched, method = _select_candidate(
                basename_matches,
                record,
                page_url,
                "basename",
            )

            if matched is not None:
                return matched, method

            if method == "basename_ambiguous":
                return None, method

    # --------------------------------------------------------
    # 3. unique alt fallback
    #
    # page가 수정되어 image URL만 변한 경우를 위한 최후의 보조 매칭.
    # 빈 alt에는 절대 적용하지 않는다.
    # --------------------------------------------------------

    if target_alt:
        alt_matches = [
            image
            for image in images
            if normalize_text(image.get("alt")) == target_alt
        ]

        if len(alt_matches) == 1:
            return alt_matches[0], "unique_alt"

        if len(alt_matches) > 1:
            matched, method = _select_candidate(
                alt_matches,
                record,
                page_url,
                "alt",
            )

            if matched is not None:
                return matched, method

            return None, method

    return None, "not_found"




async def inspect_matched_image(
    page,
    scan_id: str,
) -> dict | None:
    """
    찾은 이미지가 어떤 <a> 안에 있고,
    같은 링크에 visible text가 있는지 검사한다.

    v2:
    이미지 크기/클래스도 같이 수집해서
    tiny UI icon과 thumbnail을 구분하는 데 사용한다.
    """

    locator = page.locator(
        f'img[data-koaltq-scan-id="{scan_id}"]'
    )

    if await locator.count() == 0:
        return None

    try:
        return await locator.first.evaluate(
            """
            img => {
                const anchor = img.closest("a[href]");
                const rect = img.getBoundingClientRect();

                const base = {
                    in_header: !!img.closest("header"),
                    in_nav: !!img.closest("nav"),
                    in_footer: !!img.closest("footer"),
                    img_alt: img.getAttribute("alt") || "",
                    img_current_src: img.currentSrc || img.src || "",
                    img_class: img.className || "",
                    rendered_width: Math.round(rect.width || 0),
                    rendered_height: Math.round(rect.height || 0),
                    natural_width: img.naturalWidth || 0,
                    natural_height: img.naturalHeight || 0
                };

                if (!anchor) {
                    return {
                        ...base,
                        anchor_found: false,
                        anchor_text: "",
                        anchor_href: "",
                        anchor_class: ""
                    };
                }

                const text = (anchor.innerText || "")
                    .replace(/\\s+/g, " ")
                    .trim();

                return {
                    ...base,
                    anchor_found: true,
                    anchor_text: text,
                    anchor_href: anchor.href || "",
                    anchor_class: anchor.className || ""
                };
            }
            """
        )

    except Exception:
        return None


# ============================================================
# Thumbnail rule
# ============================================================

def decide_thumbnail(
    record: dict,
    dom: dict | None,
) -> tuple[bool | None, str]:
    """
    v0.9의 썸네일 카드 정의를 DOM 기반으로 구현.

    최종 규칙:
      1) 이미지 + 같은 링크의 제목/설명 텍스트가 있어야 한다.
      2) header/nav/footer, image-only link, 일반 CTA는 제외한다.
      3) 단, 작은 아이콘/그래픽이어도 alt 핵심 내용이 같은 링크의
         제목 텍스트에 반복되면 graphic thumbnail로 인정한다.
      4) 위 반복 관계가 없을 때만 icon/logo/small UI 규칙으로 제외한다.
    """

    if not dom:
        return None, "dom_inspection_failed"

    if not dom.get("anchor_found"):
        return None, "in_link_true_but_anchor_not_found"

    anchor_text = normalize_text(dom.get("anchor_text"))

    # --------------------------------------------------------
    # 사이트 공통 UI는 썸네일 카드가 아님
    # --------------------------------------------------------
    if (
        dom.get("in_header")
        or dom.get("in_nav")
        or dom.get("in_footer")
    ):
        return False, "site_chrome_link"

    # --------------------------------------------------------
    # 이미지가 링크의 유일 콘텐츠면 thumbnail card가 아님
    # --------------------------------------------------------
    if not anchor_text:
        return False, "image_only_link"

    # --------------------------------------------------------
    # 같은 링크의 텍스트가 단순 CTA/UI 문구면 제목이 아님
    # --------------------------------------------------------
    if not has_meaningful_text(anchor_text):
        return False, "non_title_link_text"

    # --------------------------------------------------------
    # SMALL/ICON 예외를 먼저 검사
    #
    # 예) alt='교통과 이미지', anchor='교통과 교통과'
    #     -> 작은 이미지여도 graphic thumbnail=True
    #
    # 반면
    # alt='개인정보처리 아이콘 이미지',
    # anchor='제9조 (개인정보 보호책임자)'
    #     -> 내용 반복이 없으므로 아래 UI/icon 필터로 False
    # --------------------------------------------------------
    if alt_repeats_link_title(
        record.get("alt_text"),
        anchor_text,
    ):
        return True, "alt_repeats_link_title"

    # --------------------------------------------------------
    # 반복 관계가 없는 icon/logo/file-format/small UI 제외
    # --------------------------------------------------------
    non_thumbnail, non_thumbnail_reason = obvious_non_thumbnail_visual(
        record,
        dom,
    )

    if non_thumbnail:
        return False, non_thumbnail_reason or "non_thumbnail_visual"

    # --------------------------------------------------------
    # 일반적인 콘텐츠 카드
    # --------------------------------------------------------
    return True, "image_and_title_text_in_same_link"


# ============================================================
# Page processing
# ============================================================

async def load_page(page, url: str) -> tuple[bool, str]:
    """
    페이지 접속. 최대 2회 시도.
    """

    last_error = ""

    for attempt in range(2):
        try:
            response = await page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=25_000,
            )

            await page.wait_for_timeout(1000)

            if response is not None and response.status >= 400:
                last_error = f"http_{response.status}"

                if attempt == 0:
                    continue

                return False, last_error

            return True, "success"

        except Exception as e:
            last_error = type(e).__name__

            if attempt == 0:
                await page.wait_for_timeout(1000)
                continue

    return False, last_error or "unknown_error"


async def process(
    input_path: Path,
    output_path: Path,
    audit_path: Path,
    limit: int | None,
    headful: bool,
    delay: float,
) -> None:
    paths = [Path(p).resolve() for p in (input_path, output_path, audit_path)]
    if len(set(paths)) != 3:
        raise ValueError("Input, output and audit must be separate files")
    if output_path.exists() or audit_path.exists():
        raise FileExistsError("Output/audit already exists; preserve the historical features or choose new paths")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if not 0 <= delay < float("inf"):
        raise ValueError("delay must be finite and nonnegative")
    records = load_jsonl(input_path)
    ids = [r.get("record_id") for r in records]
    if any(not isinstance(rid, str) or not rid for rid in ids) or len(set(ids)) != len(ids):
        raise ValueError("record_id must be nonempty and unique")
    from playwright.async_api import async_playwright

    if limit is not None:
        records = records[:limit]

    print(f"Loaded records: {len(records)}")

    thumbnail_results: dict[str, bool | None] = {}
    audit_results: dict[str, dict] = {}

    # --------------------------------------------------------
    # in_link=False는 웹을 방문할 필요가 없음
    # --------------------------------------------------------

    page_groups: dict[str, list[dict]] = defaultdict(list)

    for record in records:
        record_id = record["record_id"]

        if not record.get("in_link", False):
            thumbnail_results[record_id] = False

            audit_results[record_id] = {
                "record_id": record_id,
                "page_url": record.get("page_url", ""),
                "image_url": record.get("image_url", ""),
                "in_link": False,
                "page_status": "not_requested",
                "image_match_method": None,
                "decision_source": "dataset_in_link",
                "anchor_found": None,
                "anchor_text": "",
                "dataset_link_dest": normalize_text(record.get("link_dest")),
                "thumbnail": False,
                "reason": "in_link_false",
            }

        else:
            page_url = normalize_text(
                record.get("page_url")
            )

            if not page_url:
                thumbnail_results[record_id] = None

                audit_results[record_id] = {
                    "record_id": record_id,
                    "page_url": "",
                    "image_url": record.get("image_url", ""),
                    "in_link": True,
                    "page_status": "missing_page_url",
                    "image_match_method": None,
                    "decision_source": "unknown",
                    "anchor_found": None,
                    "anchor_text": "",
                    "dataset_link_dest": normalize_text(record.get("link_dest")),
                    "thumbnail": None,
                    "reason": "missing_page_url",
                }

            else:
                page_groups[page_url].append(record)

    print(
        f"Pages requiring DOM inspection: {len(page_groups)}"
    )

    # --------------------------------------------------------
    # Browser
    # --------------------------------------------------------

    async with async_playwright() as p:

        browser = await p.chromium.launch(
            headless=not headful,
        )

        context = await browser.new_context(
            locale="ko-KR",
            ignore_https_errors=True,
            viewport={
                "width": 1440,
                "height": 1200,
            },
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/130.0 Safari/537.36"
            ),
        )

        page = await context.new_page()
        page.set_default_timeout(7000)

        total_pages = len(page_groups)

        for page_index, (page_url, group) in enumerate(
            page_groups.items(),
            start=1,
        ):

            print(
                f"[{page_index}/{total_pages}] "
                f"{page_url} "
                f"({len(group)} records)"
            )

            success, page_status = await load_page(
                page,
                page_url,
            )

            # ------------------------------------------------
            # 페이지 자체 접속 실패
            # -> 이전 설계대로 null 유지
            # ------------------------------------------------

            if not success:
                for record in group:
                    record_id = record["record_id"]

                    thumbnail_results[record_id] = None

                    audit_results[record_id] = {
                        "record_id": record_id,
                        "page_url": page_url,
                        "image_url": record.get("image_url", ""),
                        "in_link": True,
                        "page_status": page_status,
                        "image_match_method": None,
                        "decision_source": "unknown",
                        "anchor_found": None,
                        "anchor_text": "",
                        "dataset_link_dest": normalize_text(record.get("link_dest")),
                        "thumbnail": None,
                        "reason": "page_fetch_failed",
                    }

                await asyncio.sleep(delay)
                continue

            # ------------------------------------------------
            # 현재 DOM image 목록
            # ------------------------------------------------

            images = await collect_images(page)
            scrolled = False

            for record in group:

                record_id = record["record_id"]

                matched, match_method = match_image(
                    images,
                    record,
                    page.url,
                )

                # --------------------------------------------
                # lazy loading 가능성 때문에 한 번만 scroll 후 재탐색
                # --------------------------------------------

                if matched is None and not scrolled:
                    await auto_scroll(page)

                    images = await collect_images(page)
                    scrolled = True

                    matched, match_method = match_image(
                        images,
                        record,
                        page.url,
                    )

                # --------------------------------------------
                # 이미지 자체를 못 찾음
                #
                # v2 핵심:
                # None으로 버리지 않고 link_dest metadata fallback 사용
                # --------------------------------------------

                if matched is None:
                    fallback_thumbnail, fallback_reason = (
                        fallback_thumbnail_from_metadata(record)
                    )

                    thumbnail_results[record_id] = fallback_thumbnail

                    audit_results[record_id] = {
                        "record_id": record_id,
                        "page_url": page_url,
                        "image_url": record.get("image_url", ""),
                        "in_link": True,
                        "page_status": "success",
                        "image_match_method": match_method,
                        "decision_source": "metadata_fallback",
                        "anchor_found": None,
                        "anchor_text": "",
                        "dataset_link_dest": normalize_text(
                            record.get("link_dest")
                        ),
                        "thumbnail": fallback_thumbnail,
                        "reason": fallback_reason,
                    }

                    continue

                # --------------------------------------------
                # 찾은 img의 link 구조 조사
                # --------------------------------------------

                dom = await inspect_matched_image(
                    page,
                    matched["scan_id"],
                )

                thumbnail, reason = decide_thumbnail(
                    record,
                    dom,
                )

                thumbnail_results[record_id] = thumbnail

                anchor_text = (
                    normalize_text(
                        dom.get("anchor_text")
                    )
                    if dom
                    else ""
                )

                dataset_link_dest = normalize_text(
                    record.get("link_dest")
                )

                audit_results[record_id] = {
                    "record_id": record_id,
                    "page_url": page_url,
                    "image_url": record.get("image_url", ""),
                    "in_link": True,
                    "page_status": "success",
                    "image_match_method": match_method,
                    "decision_source": (
                        "dom"
                        if thumbnail is not None
                        else "unknown"
                    ),
                    "dom_img_alt": (
                        dom.get("img_alt", "")
                        if dom
                        else ""
                    ),
                    "dom_img_class": (
                        dom.get("img_class", "")
                        if dom
                        else ""
                    ),
                    "rendered_width": (
                        dom.get("rendered_width")
                        if dom
                        else None
                    ),
                    "rendered_height": (
                        dom.get("rendered_height")
                        if dom
                        else None
                    ),
                    "natural_width": (
                        dom.get("natural_width")
                        if dom
                        else None
                    ),
                    "natural_height": (
                        dom.get("natural_height")
                        if dom
                        else None
                    ),
                    "anchor_found": (
                        dom.get("anchor_found")
                        if dom
                        else None
                    ),
                    "anchor_href": (
                        dom.get("anchor_href", "")
                        if dom
                        else ""
                    ),
                    "anchor_class": (
                        dom.get("anchor_class", "")
                        if dom
                        else ""
                    ),
                    "anchor_text": anchor_text,
                    "dataset_link_dest": dataset_link_dest,
                    "anchor_eq_link_dest": (
                        normalize_text(anchor_text)
                        == normalize_text(dataset_link_dest)
                        if anchor_text and dataset_link_dest
                        else False
                    ),
                    "in_header": (
                        dom.get("in_header")
                        if dom
                        else None
                    ),
                    "in_nav": (
                        dom.get("in_nav")
                        if dom
                        else None
                    ),
                    "in_footer": (
                        dom.get("in_footer")
                        if dom
                        else None
                    ),
                    "thumbnail": thumbnail,
                    "reason": reason,
                }

            await asyncio.sleep(delay)

        await context.close()
        await browser.close()

    # ========================================================
    # 원본 레코드 순서 유지 + thumbnail 하나만 추가
    # ========================================================

    output_records = []

    for record in records:
        new_record = dict(record)

        new_record["thumbnail"] = thumbnail_results.get(
            record["record_id"]
        )

        output_records.append(new_record)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        output_path,
        output_records,
    )

    # audit 파일도 원본 순서로 저장
    ordered_audit = [
        audit_results[record["record_id"]]
        for record in records
    ]

    write_jsonl(
        audit_path,
        ordered_audit,
    )

    # ========================================================
    # 통계 출력
    # ========================================================

    counts = Counter(
        record["thumbnail"]
        for record in output_records
    )

    reasons = Counter(
        row["reason"]
        for row in ordered_audit
    )

    sources = Counter(
        row.get("decision_source")
        for row in ordered_audit
    )

    print()
    print("=" * 60)
    print("Finished")
    print("=" * 60)

    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print(f"Audit : {audit_path}")

    print()
    print("Thumbnail counts")
    print("----------------")

    print(f"True   : {counts.get(True, 0)}")
    print(f"False  : {counts.get(False, 0)}")
    print(f"Unknown: {counts.get(None, 0)}")

    print()
    print("Decision sources")
    print("----------------")

    for source, count in sources.most_common():
        print(f"{str(source):40s} {count}")

    print()
    print("Reasons")
    print("-------")

    for reason, count in reasons.most_common():
        print(f"{reason:40s} {count}")


# ============================================================
# CLI
# ============================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Defaults to records_with_thumbnail.jsonl next to --input",
    )

    parser.add_argument(
        "--audit",
        type=Path,
        help="Defaults to thumbnail_audit.jsonl next to --input",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="처음 N개 record만 시험 실행",
    )

    parser.add_argument(
        "--headful",
        action="store_true",
        help="Chromium 창을 실제로 띄움",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="page_url별 요청 사이 대기 시간(초)",
    )

    args = parser.parse_args(argv)
    args.output = args.output or args.input.with_name("records_with_thumbnail.jsonl")
    args.audit = args.audit or args.input.with_name("thumbnail_audit.jsonl")
    return args


def main():
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {args.input}"
        )

    asyncio.run(process(
        input_path=args.input,
        output_path=args.output,
        audit_path=args.audit,
        limit=args.limit,
        headful=args.headful,
        delay=args.delay,
    ))


if __name__ == "__main__":
    main()
