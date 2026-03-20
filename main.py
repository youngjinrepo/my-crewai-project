import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM

# 환경 변수 로드
load_dotenv()

# Gemini LLM 설정
llm = LLM(
    model="gemini-2.5-flash",
    provider="google",
    api_key=os.getenv("GOOGLE_API_KEY")
)

# 에이전트 정의
analyst = Agent(
    role="시니어 기술 분석가",
    goal="AI 기술 트렌드를 구조적으로 분석한다",
    backstory="10년 경력의 AI 산업 및 시장 분석 전문가",
    llm=llm,
    verbose=True
)

# 작업 정의 (expected_output 필수)
task = Task(
    description="AI 에이전트 시스템의 현재 기술 수준과 2026년 전망을 3가지 핵심 포인트로 요약해줘.",
    expected_output="3가지 핵심 포인트로 구성된 명확한 분석 요약",
    agent=analyst
)

# Crew 구성
crew = Crew(
    agents=[analyst],
    tasks=[task],
    verbose=True
)

# 실행
if __name__ == "__main__":
    result = crew.kickoff()
    print("\n\n===== 결과 =====\n")
    print(result)
