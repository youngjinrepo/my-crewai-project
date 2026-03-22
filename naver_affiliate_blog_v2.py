import argparse
import html
import json
import os
import random
import re
import sys
import time
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from textwrap import dedent
from urllib.parse import urlparse

import requests
from crewai import Agent, Crew, LLM, Process, Task
from crewai.memory.storage.kickoff_task_outputs_storage import KickoffTaskOutputsSQLiteStorage
from dotenv import load_dotenv

CREWAI_STATE_ROOT = Path(os.getenv("CREWAI_STATE_ROOT", ".crewai_state")).resolve()
CREWAI_STATE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["LOCALAPPDATA"] = str(CREWAI_STATE_ROOT)
os.environ["APPDATA"] = str(CREWAI_STATE_ROOT)
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

DEFAULT_MODEL = os.getenv("BLOG_LLM_MODEL", "gemini-2.5-flash")
DEFAULT_OUTPUT_DIR = Path(os.getenv("BLOG_OUTPUT_DIR", "outputs"))
DEFAULT_DISCLOSURE = (
    "본 포스팅은 네이버 브랜드 커넥트 제휴 마케팅 활동의 일환으로 "
    "수수료를 제공받을 수 있습니다."
)
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/135.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}
PRODUCT_TEMPLATES = {
    "auto": {
        "label": "Auto detect",
        "audience_hint": "Infer the strongest women-led buyer segment from the product itself.",
        "pain_points": [
            "What practical problem this product solves in daily life",
            "Why the reader might hesitate before buying",
            "What would make the product feel trustworthy",
        ],
        "trust_points": [
            "Real-life usability",
            "Balanced pros and cons",
            "Readable and practical tone",
        ],
        "writing_angle": "Lead with usefulness, fit, and realistic purchase motivation.",
    },
    "beauty": {
        "label": "Beauty",
        "audience_hint": "Focus on women in their 20s to 40s and sensitive middle-aged buyers when relevant.",
        "pain_points": [
            "Skin concerns, irritation worries, ingredient curiosity",
            "Fit for daily routine and makeup layering",
            "Value versus premium perception",
        ],
        "trust_points": [
            "Texture, finish, ingredient or formula positioning",
            "Daily usability and skin-type fit",
            "Soft, credible tone without miracle claims",
        ],
        "writing_angle": "Write like a calm beauty blogger explaining texture, usage context, and who it fits.",
    },
    "health_food": {
        "label": "Health food",
        "audience_hint": "Focus on women in their 30s to middle-aged readers who care about routine and family wellness.",
        "pain_points": [
            "Need for easy wellness habits",
            "Taste, routine burden, and repurchase hesitation",
            "Concern about exaggerated health claims",
        ],
        "trust_points": [
            "Daily intake convenience",
            "Ingredient and consumption context",
            "Cautious wording with no medical certainty",
        ],
        "writing_angle": "Use a routine-and-lifestyle angle, never a treatment or cure angle.",
    },
    "living": {
        "label": "Living",
        "audience_hint": "Focus on women in their 30s to 50s managing home comfort, cleanup, or organization.",
        "pain_points": [
            "Household inconvenience repeated every day",
            "Need for space-saving, cleanup, or comfort",
            "Worry about whether the product is actually worth the money",
        ],
        "trust_points": [
            "Convenience in real spaces",
            "Before-and-after daily routine improvement",
            "Practical and grounded wording",
        ],
        "writing_angle": "Make the product feel like a realistic quality-of-life upgrade.",
    },
    "fashion": {
        "label": "Fashion",
        "audience_hint": "Focus on women in their 20s to 40s who care about styling versatility and wearability.",
        "pain_points": [
            "How to style it without effort",
            "Fit, comfort, and value concerns",
            "Need for wearable looks beyond one occasion",
        ],
        "trust_points": [
            "Styling flexibility",
            "Silhouette, material feel, and seasonality",
            "Confident but not overhyped tone",
        ],
        "writing_angle": "Write like a practical styling guide rather than a hard sell.",
    },
    "digital": {
        "label": "Digital",
        "audience_hint": "Focus on women in their 20s to 40s and middle-aged readers looking for ease, clarity, and reliability.",
        "pain_points": [
            "Feature overload and comparison fatigue",
            "Need for easy setup or daily convenience",
            "Concern about price versus actual usefulness",
        ],
        "trust_points": [
            "Simple benefit explanation",
            "Use-case based comparisons",
            "Low-jargon, decision-friendly language",
        ],
        "writing_angle": "Explain features through everyday scenarios, not spec dumping.",
    },
}


class SimpleHTMLCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self.meta = {}
        self.ld_json_blocks = []
        self.script_blocks = []
        self._capture_title = False
        self._capture_script = False
        self._script_type = ""
        self._buffer = []

    def handle_starttag(self, tag, attrs):
        attr_map = dict(attrs)
        if tag == "title":
            self._capture_title = True
            self._buffer = []
        elif tag == "meta":
            key = attr_map.get("property") or attr_map.get("name")
            value = attr_map.get("content")
            if key and value:
                self.meta[key.lower()] = value.strip()
        elif tag == "script":
            self._capture_script = True
            self._script_type = attr_map.get("type", "").lower()
            self._buffer = []

    def handle_endtag(self, tag):
        if tag == "title" and self._capture_title:
            self.title = "".join(self._buffer).strip()
            self._capture_title = False
        elif tag == "script" and self._capture_script:
            content = "".join(self._buffer).strip()
            if content:
                if self._script_type == "application/ld+json":
                    self.ld_json_blocks.append(content)
                else:
                    self.script_blocks.append(content)
            self._capture_script = False
            self._script_type = ""

    def handle_data(self, data):
        if self._capture_title or self._capture_script:
            self._buffer.append(data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Naver affiliate blog package with store-page research."
    )
    parser.add_argument("--product-name", default="", help="Optional product name override")
    parser.add_argument("--product-url", required=True, help="Naver store product URL")
    parser.add_argument("--target-segment", default="", help="Optional target segment")
    parser.add_argument("--banned-expressions", nargs="*", default=[], help="Optional banned phrases")
    parser.add_argument("--product-category", choices=sorted(PRODUCT_TEMPLATES.keys()), default="auto")
    parser.add_argument("--output-format", choices=["md", "txt"], default="md")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--disclosure", default=DEFAULT_DISCLOSURE)
    parser.add_argument("--fetch-timeout", type=int, default=20)
    parser.add_argument("--max-images", type=int, default=5)
    parser.add_argument("--max-review-snippets", type=int, default=12)
    return parser.parse_args()


def build_llm(model_name: str) -> LLM:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set in the environment.")
    return LLM(model=model_name, provider="google", api_key=api_key)


def get_template(category: str) -> dict:
    return PRODUCT_TEMPLATES.get(category, PRODUCT_TEMPLATES["auto"])


def build_template_block(payload: dict) -> str:
    template = get_template(payload["product_category"])
    pain_points = "\n".join(f"- {item}" for item in template["pain_points"])
    trust_points = "\n".join(f"- {item}" for item in template["trust_points"])
    return dedent(
        f"""
        Category template:
        - Category: {payload["product_category"]} ({template["label"]})
        - Audience hint: {template["audience_hint"]}
        - Writing angle: {template["writing_angle"]}
        - Pain point hints:
        {pain_points}
        - Trust point hints:
        {trust_points}
        """
    ).strip()


def fetch_url(url: str, timeout: int) -> str:
    session = requests.Session()
    parsed = urlparse(url)
    base_url = f"{parsed.scheme}://{parsed.netloc}"
    headers = {
        **DEFAULT_HEADERS,
        "Referer": base_url + "/",
        "Origin": base_url,
    }

    last_error = None
    for attempt in range(4):
        try:
            response = session.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                sleep_seconds = float(retry_after) if retry_after and retry_after.isdigit() else (1.5 * (attempt + 1))
                time.sleep(sleep_seconds + random.uniform(0.2, 0.8))
                continue
            response.raise_for_status()
            response.encoding = response.apparent_encoding or response.encoding
            return response.text
        except requests.HTTPError as exc:
            last_error = exc
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code in {403, 429} and attempt < 3:
                time.sleep((1.5 * (attempt + 1)) + random.uniform(0.2, 0.8))
                continue
            break
        except requests.RequestException as exc:
            last_error = exc
            if attempt < 3:
                time.sleep((1.0 * (attempt + 1)) + random.uniform(0.2, 0.8))
                continue
            break

    raise RuntimeError(
        "Failed to fetch the product page. Naver may be rate-limiting or blocking automated requests. "
        "Try again later, reduce repeated runs, or open the page manually and pass the key facts into v1."
    ) from last_error


def safe_json_loads(raw: str):
    try:
        return json.loads(raw)
    except Exception:
        return None


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(value or "")).strip()


def walk_json(node, callback):
    if isinstance(node, dict):
        callback(node)
        for value in node.values():
            walk_json(value, callback)
    elif isinstance(node, list):
        for item in node:
            walk_json(item, callback)


def looks_like_review_text(text: str) -> bool:
    normalized = normalize_whitespace(text)
    if len(normalized) < 12 or len(normalized) > 220:
        return False
    markers = ["배송", "재구매", "만족", "아쉬", "편해", "좋아요", "사용", "착용", "포장"]
    return any(marker in normalized for marker in markers)


def extract_price_candidates(text: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"(?<!\d)(\d{1,3}(?:,\d{3})+)\s*원", text)))


def extract_image_urls(text: str, max_images: int) -> list[str]:
    matches = re.findall(r'https://[^"\']+\.(?:jpg|jpeg|png|webp)(?:\?[^"\']*)?', text, re.IGNORECASE)
    return list(dict.fromkeys([url for url in matches if "naver" in url or "phinf" in url]))[:max_images]


def extract_ld_json_product(blocks: list[str]) -> dict:
    result = {"title": "", "description": "", "brand": "", "price": "", "images": []}
    for block in blocks:
        data = safe_json_loads(block)
        if not data:
            continue
        nodes = data if isinstance(data, list) else [data]
        for node in nodes:
            if not isinstance(node, dict) or node.get("@type") != "Product":
                continue
            result["title"] = result["title"] or normalize_whitespace(node.get("name", ""))
            result["description"] = result["description"] or normalize_whitespace(node.get("description", ""))
            brand = node.get("brand")
            if isinstance(brand, dict):
                result["brand"] = result["brand"] or normalize_whitespace(brand.get("name", ""))
            elif isinstance(brand, str):
                result["brand"] = result["brand"] or normalize_whitespace(brand)
            images = node.get("image") or []
            if isinstance(images, str):
                images = [images]
            result["images"].extend(images)
            offers = node.get("offers")
            if isinstance(offers, dict):
                result["price"] = result["price"] or str(offers.get("price", "")).strip()
    result["images"] = list(dict.fromkeys([img for img in result["images"] if img]))
    return result


def extract_script_facts(script_blocks: list[str], max_review_snippets: int, max_images: int) -> dict:
    review_snippets = []
    feature_candidates = []
    price_candidates = []
    image_urls = []
    for block in script_blocks:
        price_candidates.extend(extract_price_candidates(block))
        image_urls.extend(extract_image_urls(block, max_images))
        direct_review_hits = re.findall(
            r'"(?:reviewContent|reviewText|content)"\s*:\s*"([^"]+)"',
            block,
            re.IGNORECASE,
        )
        for hit in direct_review_hits:
            cleaned = normalize_whitespace(hit)
            if looks_like_review_text(cleaned):
                review_snippets.append(cleaned)
        if "review" not in block.lower() and "상품" not in block and "price" not in block.lower():
            continue
        candidates = re.findall(r"(\{.*?\}|\[.*?\])", block, re.DOTALL)
        for candidate in candidates[:20]:
            data = safe_json_loads(candidate)
            if data is None:
                continue

            def collect(node):
                if not isinstance(node, dict):
                    return
                for key, value in node.items():
                    key_lower = str(key).lower()
                    if isinstance(value, str):
                        cleaned = normalize_whitespace(value)
                        if key_lower in {"reviewcontent", "content", "reviewtext", "reviewcontenttext"} and looks_like_review_text(cleaned):
                            review_snippets.append(cleaned)
                        if key_lower in {"summary", "description", "title", "headline", "benefit"} and 8 <= len(cleaned) <= 140:
                            feature_candidates.append(cleaned)

            walk_json(data, collect)
    return {
        "review_snippets": list(dict.fromkeys(review_snippets))[:max_review_snippets],
        "feature_candidates": list(dict.fromkeys(feature_candidates))[:12],
        "price_candidates": list(dict.fromkeys(price_candidates))[:5],
        "image_urls": list(dict.fromkeys(image_urls))[:max_images],
    }


def extract_visible_features(html_text: str) -> list[str]:
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", html_text)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", "\n", text)
    text = normalize_whitespace(text)
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    features = []
    for part in parts:
        cleaned = normalize_whitespace(part)
        if 12 <= len(cleaned) <= 160 and not cleaned.startswith("http"):
            features.append(cleaned)
    return list(dict.fromkeys(features))


def summarize_store_page(url: str, html_text: str, max_images: int, max_review_snippets: int) -> dict:
    parser = SimpleHTMLCollector()
    parser.feed(html_text)
    ld_product = extract_ld_json_product(parser.ld_json_blocks)
    script_facts = extract_script_facts(parser.script_blocks, max_review_snippets, max_images)
    visible_features = extract_visible_features(html_text)
    page_title = normalize_whitespace(parser.meta.get("og:title") or parser.title)
    description = normalize_whitespace(parser.meta.get("description") or parser.meta.get("og:description") or ld_product["description"])
    images = ld_product["images"] + ([parser.meta.get("og:image")] if parser.meta.get("og:image") else []) + script_facts["image_urls"]
    features = [description] + script_facts["feature_candidates"] + visible_features[:20]
    price_candidates = ([ld_product["price"]] if ld_product["price"] else []) + extract_price_candidates(html_text) + script_facts["price_candidates"]
    return {
        "source_url": url,
        "source_host": urlparse(url).netloc,
        "page_title": page_title,
        "product_name": ld_product["title"] or page_title,
        "brand": ld_product["brand"],
        "description": description,
        "price_candidates": list(dict.fromkeys([item for item in price_candidates if item]))[:5],
        "image_urls": list(dict.fromkeys([item for item in images if item]))[:max_images],
        "review_snippets": script_facts["review_snippets"],
        "feature_candidates": list(dict.fromkeys([item for item in features if 12 <= len(item) <= 200]))[:12],
        "scrape_notes": [
            "This research is heuristic and based on HTML metadata, visible text, and embedded script data.",
            "Review snippets may be partial if the page loads review data dynamically.",
            "Image OCR is not performed in v2.",
        ],
    }


def build_fallback_research(args: argparse.Namespace, error: Exception | None = None) -> dict:
    note = (
        f"Store-page fetch failed: {error}" if error else "Store-page fetch was skipped."
    )
    return {
        "source_url": args.product_url.strip(),
        "source_host": urlparse(args.product_url.strip()).netloc,
        "page_title": "",
        "product_name": args.product_name.strip(),
        "brand": "",
        "description": "",
        "price_candidates": [],
        "image_urls": [],
        "review_snippets": [],
        "feature_candidates": [],
        "scrape_notes": [
            note,
            "V2 automatically fell back to a no-research mode similar to v1.",
            "Add a product name manually for better output when the page cannot be fetched.",
        ],
    }


def build_input_payload(args: argparse.Namespace, research: dict) -> dict:
    product_name = args.product_name.strip() or research.get("product_name", "")
    return {
        "product_name": product_name,
        "product_url": args.product_url.strip(),
        "product_category": args.product_category,
        "target_segment": args.target_segment.strip(),
        "banned_expressions": [expr.strip() for expr in args.banned_expressions if expr.strip()],
        "output_format": args.output_format,
        "disclosure": args.disclosure.strip(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "research": research,
    }


def build_research_block(payload: dict) -> str:
    research = payload["research"]
    feature_lines = "\n".join(f"- {item}" for item in research.get("feature_candidates", [])[:10]) or "- None found"
    review_lines = "\n".join(f"- {item}" for item in research.get("review_snippets", [])[:10]) or "- None found"
    image_lines = "\n".join(f"- {item}" for item in research.get("image_urls", [])[:5]) or "- None found"
    note_lines = "\n".join(f"- {item}" for item in research.get("scrape_notes", []))
    price_line = ", ".join(research.get("price_candidates", [])) or "None found"
    return dedent(
        f"""
        Store page research:
        - Page title: {research.get("page_title", "")}
        - Product name from page: {research.get("product_name", "")}
        - Brand: {research.get("brand", "")}
        - Description: {research.get("description", "")}
        - Price candidates: {price_line}
        - Feature candidates:
        {feature_lines}
        - Review snippets:
        {review_lines}
        - Image URLs:
        {image_lines}
        - Research notes:
        {note_lines}
        """
    ).strip()


def build_strategy_prompt(payload: dict) -> str:
    banned_text = ", ".join(payload["banned_expressions"]) or "None"
    target_segment = payload["target_segment"] or "Infer from product fit and store-page evidence"
    return dedent(
        f"""
        You are a Naver blog affiliate content strategist.
        Create a Korean JSON blog brief from the input below.

        Input:
        - Product name: {payload["product_name"]}
        - Product URL: {payload["product_url"]}
        - Product category: {payload["product_category"]}
        - Optional target segment: {target_segment}
        - Banned expressions: {banned_text}

        {build_template_block(payload)}

        {build_research_block(payload)}

        Requirements:
        - Base the strategy on store-page evidence first, not guesswork.
        - Use review snippets as user sentiment clues, but keep wording careful.
        - Avoid exaggerated claims, medical certainty, and unsupported ranking language.
        - If evidence is incomplete, stay cautious and avoid inventing specs.

        Follow this JSON schema exactly:
        {{
          "core_target": "Korean text",
          "secondary_target": "Korean text",
          "tone_style": "Korean text",
          "search_intent": "Korean text",
          "seo_keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5", "keyword6"],
          "hook_angles": ["angle1", "angle2", "angle3"],
          "pain_points": ["pain1", "pain2", "pain3"],
          "trust_points": ["trust1", "trust2", "trust3"],
          "cta_angle": "Korean text",
          "image_copy_direction": "Korean text",
          "fact_guardrails": ["fact1", "fact2", "fact3"]
        }}

        Return JSON only. No markdown. No explanation.
        """
    ).strip()


def build_writer_prompt(payload: dict, strategy_json: str) -> str:
    banned_text = ", ".join(payload["banned_expressions"]) or "None"
    return dedent(
        f"""
        You are a Naver blog affiliate copywriter.
        The strategy JSON below is provided as context from a previous task.
        Create a Korean JSON content package ready for a Naver blog post.

        Input:
        - Product name: {payload["product_name"]}
        - Product URL: {payload["product_url"]}
        - Product category: {payload["product_category"]}
        - Banned expressions: {banned_text}
        - Bottom disclosure line: {payload["disclosure"]}

        {build_template_block(payload)}

        {build_research_block(payload)}

        Strategy JSON:
        {strategy_json}

        Writing rules:
        - Use store-page facts and review evidence wherever possible.
        - If a claim is not supported by the page evidence, phrase it carefully or leave it out.
        - Generate 5 title options that are clickable but not sensational.
        - Keep subheadings short and easy to scan.
        - Write the body in a trustworthy, information-led style with short paragraphs.
        - Mention review-based impressions carefully, for example "reviewers often mention..." instead of certainty.
        - Do not repeat the URL inside the body. Use the CTA for link intent instead.
        - Write 6 image copy lines suitable for thumbnail or card-style visuals.
        - Write 12 search-friendly hashtags for Naver blog context.
        - Put the disclosure only in its own field.

        Follow this JSON schema exactly:
        {{
          "titles": ["title1", "title2", "title3", "title4", "title5"],
          "seo_keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5", "keyword6"],
          "subheadings": ["subheading1", "subheading2", "subheading3", "subheading4"],
          "intro": "Korean text",
          "body_sections": [
            {{"heading": "subheading1", "content": "Korean text"}},
            {{"heading": "subheading2", "content": "Korean text"}},
            {{"heading": "subheading3", "content": "Korean text"}}
          ],
          "closing": "Korean text",
          "cta_lines": ["CTA 1", "CTA 2", "CTA 3"],
          "image_copy": ["copy1", "copy2", "copy3", "copy4", "copy5", "copy6"],
          "hashtags": ["tag1", "tag2", "tag3", "tag4", "tag5", "tag6", "tag7", "tag8", "tag9", "tag10", "tag11", "tag12"],
          "disclosure": "{payload["disclosure"]}"
        }}

        Return JSON only. No markdown. No explanation.
        """
    ).strip()


def build_editor_prompt(payload: dict, strategy_json: str, draft_json: str) -> str:
    banned_text = ", ".join(payload["banned_expressions"]) or "None"
    return dedent(
        f"""
        You are a Naver blog editor and compliance reviewer.
        The strategy JSON and draft JSON below are provided from previous tasks.
        Review them and produce the final Korean JSON package.

        Input:
        - Product name: {payload["product_name"]}
        - Product URL: {payload["product_url"]}
        - Product category: {payload["product_category"]}
        - Banned expressions: {banned_text}
        - Bottom disclosure line: {payload["disclosure"]}

        {build_template_block(payload)}

        {build_research_block(payload)}

        Strategy JSON:
        {strategy_json}

        Draft JSON:
        {draft_json}

        Review rules:
        - Remove or soften banned expressions.
        - Reduce exaggerated or overly certain marketing claims.
        - If the draft states a fact not backed by the store research, rewrite it more carefully.
        - Keep hashtags, CTA, and image copy useful while reducing duplication.

        Follow this JSON schema exactly:
        {{
          "recommended_title": "title",
          "titles": ["title1", "title2", "title3", "title4", "title5"],
          "seo_keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5", "keyword6"],
          "subheadings": ["subheading1", "subheading2", "subheading3", "subheading4"],
          "intro": "Korean text",
          "body_sections": [
            {{"heading": "subheading1", "content": "Korean text"}},
            {{"heading": "subheading2", "content": "Korean text"}},
            {{"heading": "subheading3", "content": "Korean text"}}
          ],
          "closing": "Korean text",
          "cta_lines": ["CTA 1", "CTA 2", "CTA 3"],
          "image_copy": ["copy1", "copy2", "copy3", "copy4", "copy5", "copy6"],
          "hashtags": ["tag1", "tag2", "tag3", "tag4", "tag5", "tag6", "tag7", "tag8", "tag9", "tag10", "tag11", "tag12"],
          "disclosure": "{payload["disclosure"]}"
        }}

        Return JSON only. No markdown. No explanation.
        """
    ).strip()


def extract_json_block(raw_text: str) -> dict:
    text = raw_text.strip()
    fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced_match:
        text = fenced_match.group(1).strip()
    else:
        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last != -1 and last > first:
            text = text[first : last + 1]
    return json.loads(text)


def create_blog_crew(llm: LLM, payload: dict) -> Crew:
    strategist = Agent(role="Naver affiliate content strategist", goal="Create a strategy grounded in store-page evidence", backstory="A senior planner who turns page facts and review signals into practical content angles", llm=llm, verbose=True)
    writer = Agent(role="Naver blog copywriter", goal="Write a trustworthy Naver blog package using page evidence and review context", backstory="A copywriter who balances SEO, readability, and cautious product phrasing", llm=llm, verbose=True)
    editor = Agent(role="Naver blog editor", goal="Polish the final draft while keeping unsupported claims out", backstory="An editor who tightens affiliate writing and keeps it grounded in evidence", llm=llm, verbose=True)
    strategy_task = Task(description=build_strategy_prompt(payload), expected_output="JSON only", agent=strategist)
    draft_task = Task(description=("Read the previous task output from context first and use it as the strategy JSON.\n\n" + build_writer_prompt(payload, "Use the strategy JSON provided in context.")), expected_output="JSON only", agent=writer, context=[strategy_task])
    final_task = Task(description=("Read the previous task outputs from context first and use them as the strategy JSON and draft JSON.\n\n" + build_editor_prompt(payload, "Use the strategy JSON provided in context.", "Use the draft JSON provided in context.")), expected_output="JSON only", agent=editor, context=[strategy_task, draft_task])
    crew = Crew(agents=[strategist, writer, editor], tasks=[strategy_task, draft_task, final_task], process=Process.sequential, verbose=True)
    crew._task_output_handler.storage = KickoffTaskOutputsSQLiteStorage(db_path=str(CREWAI_STATE_ROOT / "latest_kickoff_task_outputs_v2.db"))
    return crew


def normalize_output(result) -> str:
    if hasattr(result, "raw") and result.raw:
        return result.raw
    return str(result)


def render_markdown(payload: dict, package: dict) -> str:
    research = payload["research"]
    lines = [
        f"# {package['recommended_title']}",
        "",
        "## Product Info",
        "",
        f"- Category: {payload['product_category']}",
        f"- URL: {payload['product_url']}",
        f"- Page title: {research.get('page_title', '')}",
        f"- Price candidates: {', '.join(research.get('price_candidates') or []) or 'N/A'}",
        "",
        "## Research Snapshot",
        "",
        "### Feature Candidates",
        "",
    ]
    lines.extend(f"- {item}" for item in (research.get("feature_candidates") or [])[:8])
    lines.extend(["", "### Review Snippets", ""])
    lines.extend(f"- {item}" for item in (research.get("review_snippets") or [])[:8])
    lines.extend(["", "## Title Options", ""])
    lines.extend(f"- {title}" for title in package["titles"])
    lines.extend(["", "## SEO Keywords", "", ", ".join(package["seo_keywords"]), "", "## Subheadings", ""])
    lines.extend(f"- {heading}" for heading in package["subheadings"])
    lines.extend(["", "## Body", "", package["intro"], ""])
    for section in package["body_sections"]:
        lines.extend([f"### {section['heading']}", "", section["content"], ""])
    lines.extend([package["closing"], "", "## CTA", ""])
    lines.extend(f"- {cta}" for cta in package["cta_lines"])
    lines.extend(["", "## Image Copy", ""])
    lines.extend(f"- {copy}" for copy in package["image_copy"])
    lines.extend(["", "## Hashtags", ""])
    lines.append(" ".join(tag if str(tag).startswith("#") else f"#{tag}" for tag in package["hashtags"]))
    lines.extend(["", "## Disclosure", "", package["disclosure"]])
    return "\n".join(lines).strip() + "\n"


def render_text(payload: dict, package: dict) -> str:
    research = payload["research"]
    lines = [
        f"[Recommended Title] {package['recommended_title']}",
        "",
        "[Product Info]",
        f"- Category: {payload['product_category']}",
        f"- URL: {payload['product_url']}",
        f"- Page title: {research.get('page_title', '')}",
        f"- Price candidates: {', '.join(research.get('price_candidates') or []) or 'N/A'}",
        "",
        "[Research Snapshot]",
        "- Feature candidates:",
    ]
    lines.extend(f"  {item}" for item in (research.get("feature_candidates") or [])[:8])
    lines.append("- Review snippets:")
    lines.extend(f"  {item}" for item in (research.get("review_snippets") or [])[:8])
    lines.extend(["", "[Title Options]"])
    lines.extend(f"- {title}" for title in package["titles"])
    lines.extend(["", "[SEO Keywords]", ", ".join(package["seo_keywords"]), "", "[Subheadings]"])
    lines.extend(f"- {heading}" for heading in package["subheadings"])
    lines.extend(["", "[Body]", package["intro"], ""])
    for section in package["body_sections"]:
        lines.extend([section["heading"], section["content"], ""])
    lines.extend([package["closing"], "", "[CTA]"])
    lines.extend(f"- {cta}" for cta in package["cta_lines"])
    lines.extend(["", "[Image Copy]"])
    lines.extend(f"- {copy}" for copy in package["image_copy"])
    lines.extend(["", "[Hashtags]"])
    lines.append(" ".join(tag if str(tag).startswith("#") else f"#{tag}" for tag in package["hashtags"]))
    lines.extend(["", "[Disclosure]", package["disclosure"]])
    return "\n".join(lines).strip() + "\n"


def slugify_filename(text: str) -> str:
    lowered = re.sub(r"\s+", "-", text.strip().lower())
    cleaned = re.sub(r"[^a-z0-9\-가-힣]+", "", lowered)
    return cleaned[:60] or "naver-affiliate-blog-v2"


def save_outputs(payload: dict, package: dict, output_dir: Path, output_format: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = slugify_filename(payload["product_name"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rendered = render_markdown(payload, package) if output_format == "md" else render_text(payload, package)
    extension = "md" if output_format == "md" else "txt"
    rendered_path = output_dir / f"{base_name}_v2_{timestamp}.{extension}"
    rendered_path.write_text(rendered, encoding="utf-8")
    json_path = output_dir / f"{base_name}_v2_{timestamp}.json"
    json_path.write_text(json.dumps({"input": payload, "output": package}, ensure_ascii=False, indent=2), encoding="utf-8")
    return rendered_path


def main() -> None:
    args = parse_args()
    fetch_error = None
    try:
        html_text = fetch_url(args.product_url, timeout=args.fetch_timeout)
        research = summarize_store_page(args.product_url, html_text, args.max_images, args.max_review_snippets)
    except Exception as exc:
        fetch_error = exc
        research = build_fallback_research(args, error=exc)

    payload = build_input_payload(args, research)
    llm = build_llm(args.model)
    crew = create_blog_crew(llm, payload)
    result = crew.kickoff()
    package = extract_json_block(normalize_output(result))
    output_path = save_outputs(payload, package, Path(args.output_dir), args.output_format)
    print("\n===== BLOG PACKAGE V2 READY =====\n")
    print(f"Model: {args.model}")
    if fetch_error is not None:
        print("Store-page research: fallback mode used")
        print(f"Reason: {fetch_error}")
    else:
        print("Store-page research: fetched successfully")
    print(f"Saved to: {output_path}")
    print(f"Recommended title: {package['recommended_title']}")


if __name__ == "__main__":
    main()
