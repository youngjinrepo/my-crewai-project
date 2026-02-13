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
# =========================
# 공통 컨텍스트
# =========================

COMMON_CONTEXT = """
현재 환경:

- 온프레미스
- Apache HTTP Server 유지 (사내 인증 연동 필수)
- WebLogic 일부 유지 (Java 8 기반, Legacy WAR)
- 배치 20~30개
- 배치는 SpringBoot standalone로 분리 예정
- DB는 Oracle, 서버와 분리
- 운영 인원 2명
- 클라우드 사용 불가
- 보안은 이미 강화된 상태
- 다운타임은 WAS 전환 시 최소화 필요

중요 원칙:
- 가정하지 말 것
- 불확실한 부분은 질문 목록으로 정리
- 운영 인원 2명이 감당 가능한 구조만 설계
"""

# =========================
# Agent 정의
# =========================

infra_agent = Agent(
    role="온프레미스 인프라 아키텍트",
    goal="Apache + WebLogic + SpringBoot Batch 혼합 환경의 안정적인 아키텍처 설계",
    backstory="대규모 기업 온프레미스 환경에서 WAS 전환 경험이 많은 인프라 설계 전문가",
    verbose=True
)

app_agent = Agent(
    role="레거시 Java 전환 설계 전문가",
    goal="WebLogic 기반 애플리케이션과 배치를 안전하게 분리 이전",
    backstory="레거시 WAS 기반 시스템을 Spring 기반 구조로 다수 전환한 경험 보유",
    verbose=True
)

ops_agent = Agent(
    role="전환 및 운영 전략 전문가",
    goal="다운타임 최소화 및 단순 운영 체계 수립",
    backstory="온프레미스 환경에서 최소 인원으로 시스템을 운영해온 실무 전문가",
    verbose=True
)

review_agent = Agent(
    role="기술 리스크 감사관",
    goal="설계안의 기술적 허점과 리스크를 냉정하게 지적",
    backstory="대형 장애 분석 경험이 많은 시스템 감사 전문가",
    verbose=True
)

# =========================
# Task 정의
# =========================

infra_task = Task(
    description=f"""
{COMMON_CONTEXT}

다음 산출물을 작성하라:

1. 최종 아키텍처 설계 설명
2. 프로세스 구성도 설명
3. JDK 다중 설치 전략
4. 로그 관리 구조
5. 주요 리스크 목록
6. WebLogic 장기 제거 로드맵

표 형식을 적극 활용하라.
""",
    agent=infra_agent,
    expected_output="인프라 설계 문서"
)

app_task = Task(
    description=f"""
{COMMON_CONTEXT}

다음 내용을 포함하라:

1. 배치 통합 구조 설계 (20~30개 기준)
2. WebLogic 의존성 점검 체크리스트
3. SpringBoot 전환 기준
4. Git + Jenkins 통합 전략
5. 테스트 전략

구체적이고 실행 가능한 수준으로 작성하라.
""",
    agent=app_agent,
    expected_output="애플리케이션 전환 설계 문서"
)

ops_task = Task(
    description=f"""
{COMMON_CONTEXT}

다음 산출물을 작성하라:

1. 전환 체크리스트
   - 사전 준비
   - 전환 직전
   - 전환 수행
   - 전환 직후
   - 롤백

2. DNS/IP 전환 전략
3. 장애 대응 시나리오
4. 운영 인원 2명 기준 운영 단순화 전략
5. 모니터링 구성 제안

체크리스트는 단계별 표로 작성하라.
""",
    agent=ops_agent,
    expected_output="전환 및 운영 전략 문서"
)

review_task = Task(
    description="""
위 3개의 설계 결과를 검토하라.

1. 기술적 리스크 지적
2. 빠진 고려사항
3. 과도한 설계 요소 제거 제안
4. 현실적으로 실행 어려운 부분 지적
5. 최종 보완 권고안

비판적으로 분석하라.
""",
    agent=review_agent,
    expected_output="기술 리스크 감사 보고서"
)

# =========================
# Crew 구성
# =========================

crew = Crew(
    agents=[infra_agent, app_agent, ops_agent, review_agent],
    tasks=[infra_task, app_task, ops_task, review_task],
    process=Process.sequential,
    verbose=True
)

# =========================
# 실행
# =========================

if __name__ == "__main__":
    result = crew.kickoff()
    print("\n\n===== 최종 결과 =====\n")
    print(result)