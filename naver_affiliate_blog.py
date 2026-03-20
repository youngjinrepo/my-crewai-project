import os
from datetime import datetime
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM

# Load environment variables
load_dotenv()

# LLM setup (Gemini)
llm = LLM(
    model="gemini-2.5-flash",
    provider="google",
    api_key=os.getenv("GOOGLE_API_KEY"),
)

# -----------------------------
# Input schema (edit per run)
# -----------------------------
INPUT = {
    "target_gender": "female",
    "target_age": "20s",  # e.g. "teens", "20s", "30s", "40s", "50s+"
    "season": "spring",   # e.g. "spring", "summer", "fall", "winter"
    "month": datetime.now().strftime("%B"),
    "tone": "informative",  # "informative" or "comparison"
    "price_tier": "low",     # "low", "mid", "high"
    "categories": ["beauty", "living", "digital", "lifestyle"],
    "region": "Seoul/Gyeonggi",
    "weather_summary": "",  # e.g. "Rainy for 3 days, warm afternoons"
    "weather_conditions": [],  # e.g. ["rain", "hot", "humid", "cold", "dust"]
}

# -----------------------------
# Simple heuristic recommender
# -----------------------------
SEASON_WEIGHTS = {
    "spring": {"beauty": 1.2, "living": 1.1, "digital": 0.9, "lifestyle": 1.1},
    "summer": {"beauty": 1.2, "living": 1.0, "digital": 1.0, "lifestyle": 1.1},
    "fall": {"beauty": 1.1, "living": 1.2, "digital": 1.0, "lifestyle": 1.1},
    "winter": {"beauty": 1.0, "living": 1.2, "digital": 1.1, "lifestyle": 1.0},
}

AGE_PRICE_PREF = {
    "teens": {"low": 1.2, "mid": 0.9, "high": 0.6},
    "20s": {"low": 1.2, "mid": 1.0, "high": 0.8},
    "30s": {"low": 1.0, "mid": 1.1, "high": 1.0},
    "40s": {"low": 0.9, "mid": 1.1, "high": 1.2},
    "50s+": {"low": 0.8, "mid": 1.1, "high": 1.3},
}

CATEGORY_CANDIDATES = {
    "beauty": [
        "sunscreen", "tone-up cream", "hair dryer", "facial cleansing device",
        "anti-aging serum", "scalp care set",
    ],
    "living": [
        "bedding set", "humidifier", "air purifier", "kitchen organizer",
        "mattress topper", "aroma diffuser",
    ],
    "digital": [
        "wireless earbuds", "tablet", "smartwatch", "portable SSD",
        "robot vacuum", "webcam",
    ],
    "lifestyle": [
        "water bottle", "fitness band", "travel pouch", "lunch box",
        "home workout kit", "desk lamp",
    ],
}

PURCHASE_INTENT_KEYWORDS = {
    "informative": ["best", "추천", "비교", "가성비", "구매"],
    "comparison": ["vs", "장단점", "비교", "차이", "선택"],
}

WEATHER_WEIGHTS = {
    "rain": {"living": 1.15, "lifestyle": 1.10, "digital": 1.05, "beauty": 1.00},
    "hot": {"beauty": 1.20, "lifestyle": 1.10, "living": 1.05, "digital": 1.00},
    "cold": {"living": 1.20, "beauty": 1.05, "digital": 1.05, "lifestyle": 1.00},
    "humid": {"beauty": 1.10, "living": 1.05, "lifestyle": 1.00, "digital": 1.00},
    "dust": {"living": 1.15, "beauty": 1.05, "digital": 1.00, "lifestyle": 1.00},
}


def recommend_products(payload):
    season = payload["season"]
    age = payload["target_age"]
    price = payload["price_tier"]
    categories = payload["categories"]
    conditions = payload.get("weather_conditions", [])

    season_weight = SEASON_WEIGHTS.get(season, {})
    price_weight = AGE_PRICE_PREF.get(age, {}).get(price, 1.0)

    scored = []
    for cat in categories:
        cat_weight = season_weight.get(cat, 1.0)
        base = cat_weight * price_weight
        for condition in conditions:
            base *= WEATHER_WEIGHTS.get(condition, {}).get(cat, 1.0)
        for item in CATEGORY_CANDIDATES.get(cat, []):
            scored.append((cat, item, round(base, 2)))

    # top 5 by score (simple, stable)
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:5]


def build_prompt(payload, recommendations):
    tone = payload["tone"]
    month = payload["month"]
    age = payload["target_age"]
    price = payload["price_tier"]
    region = payload.get("region", "")
    weather_summary = payload.get("weather_summary", "")
    weather_conditions = payload.get("weather_conditions", [])

    rec_lines = [f"- {cat}: {item} (score {score})" for cat, item, score in recommendations]
    weather_line = ""
    if weather_summary or weather_conditions:
        condition_str = ", ".join(weather_conditions) if weather_conditions else "none"
        weather_line = f"날씨 메모: {weather_summary} / 조건: {condition_str}\n"

    return (
        "너는 네이버 쇼핑 커넥트/브랜드 커넥트 제휴 마케팅용 블로그 글을 작성하는 "
        "전문 콘텐츠 전략가 겸 카피라이터다.\n"
        f"대상: 여성 {age}, 가격대: {price}, 시즌/월: {payload['season']} / {month}\n"
        + (f"지역: {region}\n" if region else "")
        + weather_line
        + f"톤: {tone} (정보형 또는 비교형)\n\n"
        "추천 상품 후보 (내부 추천 결과):\n"
        + "\n".join(rec_lines)
        + "\n\n"
        "아래 형식으로 결과를 작성해라:\n"
        "1) 상품 추천 Top 3~5 (각 1~2문장 요약, 왜 타겟에 맞는지)\n"
        "2) 글 아웃라인 (서론/본문/결론)\n"
        "3) 본문 (정보형 또는 비교형; 자연스럽게 구매 유도, 과장 금지)\n"
        "4) CTA 2개\n"
        "5) 해시태그 8~12개\n"
        "6) 광고 표기 문구 1개\n"
    )


# -----------------------------
# CrewAI configuration
# -----------------------------
analyst_writer = Agent(
    role="콘텐츠 전략가/카피라이터",
    goal="타겟과 시즌에 맞는 고전환 블로그 제휴 마케팅 글 작성",
    backstory="네이버 커머스 제휴 마케팅 경험이 풍부한 콘텐츠 기획자",
    llm=llm,
    verbose=True,
)

recommendations = recommend_products(INPUT)

blog_task = Task(
    description=build_prompt(INPUT, recommendations),
    expected_output="요구 형식에 맞춘 한국어 블로그 글 패키지",
    agent=analyst_writer,
)

crew = Crew(
    agents=[analyst_writer],
    tasks=[blog_task],
    verbose=True,
)


if __name__ == "__main__":
    result = crew.kickoff()
    print("\n\n===== RESULT =====\n")
    print(result)
