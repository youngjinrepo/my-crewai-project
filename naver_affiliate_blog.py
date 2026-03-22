import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from textwrap import dedent

CREWAI_STATE_ROOT = Path(os.getenv("CREWAI_STATE_ROOT", ".crewai_state")).resolve()
CREWAI_STATE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["LOCALAPPDATA"] = str(CREWAI_STATE_ROOT)
os.environ["APPDATA"] = str(CREWAI_STATE_ROOT)
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from crewai import Agent, Crew, LLM, Process, Task
from crewai.memory.storage.kickoff_task_outputs_storage import (
    KickoffTaskOutputsSQLiteStorage,
)
from dotenv import load_dotenv

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Naver affiliate blog package with CrewAI."
    )
    parser.add_argument("--product-name", required=True, help="Product name")
    parser.add_argument("--product-url", required=True, help="Product URL")
    parser.add_argument(
        "--target-segment",
        default="",
        help="Optional target segment, for example: women in their 30s",
    )
    parser.add_argument(
        "--banned-expressions",
        nargs="*",
        default=[],
        help="Optional banned phrases",
    )
    parser.add_argument(
        "--product-category",
        choices=sorted(PRODUCT_TEMPLATES.keys()),
        default="auto",
        help="Category template used to shape blog tone and hooks",
    )
    parser.add_argument(
        "--output-format",
        choices=["md", "txt"],
        default="md",
        help="Rendered file format",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="LLM model name. Default uses BLOG_LLM_MODEL or gemini-2.5-flash.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory",
    )
    parser.add_argument(
        "--disclosure",
        default=DEFAULT_DISCLOSURE,
        help="Affiliate disclosure appended at the bottom",
    )
    return parser.parse_args()


def build_llm(model_name: str) -> LLM:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set in the environment.")

    return LLM(
        model=model_name,
        provider="google",
        api_key=api_key,
    )


def build_input_payload(args: argparse.Namespace) -> dict:
    return {
        "product_name": args.product_name.strip(),
        "product_url": args.product_url.strip(),
        "product_category": args.product_category,
        "target_segment": args.target_segment.strip(),
        "banned_expressions": [expr.strip() for expr in args.banned_expressions if expr.strip()],
        "output_format": args.output_format,
        "disclosure": args.disclosure.strip(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


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


def build_strategy_prompt(payload: dict) -> str:
    banned_text = ", ".join(payload["banned_expressions"]) or "None"
    target_segment = payload["target_segment"] or "Infer from product fit"
    template_block = build_template_block(payload)

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

        {template_block}

        Requirements:
        - Consider women in their 20s, 30s, 40s, and middle-aged readers, but narrow the main buyer to 1 or 2 strongest segments.
        - Focus on purchase-intent Naver search keywords.
        - Avoid exaggerated claims, medical certainty, and unsupported "best" or "number one" phrasing.
        - Choose the tone based on the product and describe it briefly in "tone_style".
        - Design a structure that reads naturally on Naver blogs.

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
          "image_copy_direction": "Korean text"
        }}

        Return JSON only. No markdown. No explanation.
        """
    ).strip()


def build_writer_prompt(payload: dict, strategy_json: str) -> str:
    banned_text = ", ".join(payload["banned_expressions"]) or "None"
    template_block = build_template_block(payload)

    return dedent(
        f"""
        You are a Naver blog affiliate copywriter.
        The strategy JSON below is provided as context from a previous task.
        Create a Korean JSON content package that is ready for a Naver blog post.

        Input:
        - Product name: {payload["product_name"]}
        - Product URL: {payload["product_url"]}
        - Product category: {payload["product_category"]}
        - Banned expressions: {banned_text}
        - Bottom disclosure line: {payload["disclosure"]}

        {template_block}

        Strategy JSON:
        {strategy_json}

        Writing rules:
        - Generate 5 title options that are clickable but not sensational.
        - Keep subheadings short and easy to scan.
        - Write the body in a trustworthy, information-led style with short paragraphs.
        - Do not repeat the URL inside the body. Use the CTA for link intent instead.
        - Write 6 image copy lines suitable for thumbnail or card-style visuals.
        - Write 12 search-friendly hashtags for Naver blog context.
        - Put the disclosure only in its own field.
        - Match the writing angle to the selected category template.

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
    template_block = build_template_block(payload)

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

        {template_block}

        Strategy JSON:
        {strategy_json}

        Draft JSON:
        {draft_json}

        Review rules:
        - Remove or soften banned expressions.
        - Reduce exaggerated or overly certain marketing claims.
        - Improve flow if sections feel awkward.
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


def create_blog_crew(llm: LLM, payload: dict) -> tuple[Crew, Task, Task, Task]:
    strategist = Agent(
        role="Naver affiliate content strategist",
        goal="Create a Naver search-oriented blog strategy tailored to the product and buyer",
        backstory="A senior content planner who knows purchase intent and conversion-friendly blog structure",
        llm=llm,
        verbose=True,
    )

    writer = Agent(
        role="Naver blog copywriter",
        goal="Write a blog package that balances search traffic and reader retention",
        backstory="A copywriter who can naturally write informational reviews and comparison-style recommendations",
        llm=llm,
        verbose=True,
    )

    editor = Agent(
        role="Naver blog editor",
        goal="Polish the final draft with trustworthy phrasing and cleaner compliance",
        backstory="An editor who balances platform tone with affiliate marketing constraints",
        llm=llm,
        verbose=True,
    )

    strategy_task = Task(
        description=build_strategy_prompt(payload),
        expected_output="JSON only",
        agent=strategist,
    )

    draft_task = Task(
        description=(
            "Read the previous task output from context first and use it as the strategy JSON.\n\n"
            + build_writer_prompt(payload, "Use the strategy JSON provided in context.")
        ),
        expected_output="JSON only",
        agent=writer,
        context=[strategy_task],
    )

    final_task = Task(
        description=(
            "Read the previous task outputs from context first and use them as the strategy JSON and draft JSON.\n\n"
            + build_editor_prompt(
                payload,
                "Use the strategy JSON provided in context.",
                "Use the draft JSON provided in context.",
            )
        ),
        expected_output="JSON only",
        agent=editor,
        context=[strategy_task, draft_task],
    )

    crew = Crew(
        agents=[strategist, writer, editor],
        tasks=[strategy_task, draft_task, final_task],
        process=Process.sequential,
        verbose=True,
    )
    crew._task_output_handler.storage = KickoffTaskOutputsSQLiteStorage(
        db_path=str(CREWAI_STATE_ROOT / "latest_kickoff_task_outputs.db")
    )
    return crew, strategy_task, draft_task, final_task


def normalize_output(result) -> str:
    if hasattr(result, "raw") and result.raw:
        return result.raw
    return str(result)


def render_markdown(payload: dict, package: dict) -> str:
    lines = [
        f"# {package['recommended_title']}",
        "",
        "## Product Info",
        "",
        f"- Category: {payload['product_category']}",
        f"- URL: {payload['product_url']}",
        "",
        "## Title Options",
        "",
    ]
    lines.extend(f"- {title}" for title in package["titles"])
    lines.extend(
        [
            "",
            "## SEO Keywords",
            "",
            ", ".join(package["seo_keywords"]),
            "",
            "## Subheadings",
            "",
        ]
    )
    lines.extend(f"- {heading}" for heading in package["subheadings"])
    lines.extend(
        [
            "",
            "## Body",
            "",
            package["intro"],
            "",
        ]
    )

    for section in package["body_sections"]:
        lines.extend(
            [
                f"### {section['heading']}",
                "",
                section["content"],
                "",
            ]
        )

    lines.extend(
        [
            package["closing"],
            "",
            "## CTA",
            "",
        ]
    )
    lines.extend(f"- {cta}" for cta in package["cta_lines"])
    lines.extend(
        [
            "",
            "## Image Copy",
            "",
        ]
    )
    lines.extend(f"- {copy}" for copy in package["image_copy"])
    lines.extend(
        [
            "",
            "## Hashtags",
            "",
            " ".join(
                tag if str(tag).startswith("#") else f"#{tag}"
                for tag in package["hashtags"]
            ),
            "",
            "## Disclosure",
            "",
            package["disclosure"],
        ]
    )
    return "\n".join(lines).strip() + "\n"


def render_text(payload: dict, package: dict) -> str:
    lines = [
        f"[Recommended Title] {package['recommended_title']}",
        "",
        "[Product Info]",
        f"- Category: {payload['product_category']}",
        f"- URL: {payload['product_url']}",
        "",
        "[Title Options]",
    ]
    lines.extend(f"- {title}" for title in package["titles"])
    lines.extend(
        [
            "",
            "[SEO Keywords]",
            ", ".join(package["seo_keywords"]),
            "",
            "[Subheadings]",
        ]
    )
    lines.extend(f"- {heading}" for heading in package["subheadings"])
    lines.extend(
        [
            "",
            "[Body]",
            package["intro"],
            "",
        ]
    )
    for section in package["body_sections"]:
        lines.extend(
            [
                f"{section['heading']}",
                section["content"],
                "",
            ]
        )
    lines.extend(
        [
            package["closing"],
            "",
            "[CTA]",
        ]
    )
    lines.extend(f"- {cta}" for cta in package["cta_lines"])
    lines.extend(
        [
            "",
            "[Image Copy]",
        ]
    )
    lines.extend(f"- {copy}" for copy in package["image_copy"])
    lines.extend(
        [
            "",
            "[Hashtags]",
            " ".join(
                tag if str(tag).startswith("#") else f"#{tag}"
                for tag in package["hashtags"]
            ),
            "",
            "[Disclosure]",
            package["disclosure"],
        ]
    )
    return "\n".join(lines).strip() + "\n"


def slugify_filename(text: str) -> str:
    lowered = re.sub(r"\s+", "-", text.strip().lower())
    cleaned = re.sub(r"[^a-z0-9\-가-힣]+", "", lowered)
    return cleaned[:60] or "naver-affiliate-blog"


def save_outputs(payload: dict, package: dict, output_dir: Path, output_format: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = slugify_filename(payload["product_name"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    rendered = (
        render_markdown(payload, package)
        if output_format == "md"
        else render_text(payload, package)
    )
    extension = "md" if output_format == "md" else "txt"
    rendered_path = output_dir / f"{base_name}_{timestamp}.{extension}"
    rendered_path.write_text(rendered, encoding="utf-8")

    json_path = output_dir / f"{base_name}_{timestamp}.json"
    json_path.write_text(
        json.dumps(
            {
                "input": payload,
                "output": package,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return rendered_path


def main() -> None:
    args = parse_args()
    payload = build_input_payload(args)
    llm = build_llm(args.model)

    crew, _, _, _ = create_blog_crew(llm, payload)
    result = crew.kickoff()
    package = extract_json_block(normalize_output(result))

    output_path = save_outputs(
        payload=payload,
        package=package,
        output_dir=Path(args.output_dir),
        output_format=args.output_format,
    )

    print("\n===== BLOG PACKAGE READY =====\n")
    print(f"Model: {args.model}")
    print(f"Saved to: {output_path}")
    print(f"Recommended title: {package['recommended_title']}")


if __name__ == "__main__":
    main()
