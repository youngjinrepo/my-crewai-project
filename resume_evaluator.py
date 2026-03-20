import os
from pathlib import Path
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
# Input schema (file-based)
# -----------------------------
INPUT = {
    "resume_path": "resume.txt",  # 예: "resume.txt"
    "job_requirements_path": "job_requirements.txt",  # 예: "job_requirements.txt"
    "target_role": "백엔드 엔지니어",  # 예: "백엔드 엔지니어", "마케터", "PM"
}


def load_resume_text(path_str: str) -> str:
    if not path_str:
        raise ValueError("resume_path가 비어 있습니다. 파일 경로를 입력하세요.")

    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"resume_path를 찾을 수 없습니다: {path}")

    return path.read_text(encoding="utf-8").strip()


def load_job_requirements(path_str: str) -> str:
    if not path_str:
        return ""

    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"job_requirements_path를 찾을 수 없습니다: {path}")

    return path.read_text(encoding="utf-8").strip()


# -----------------------------
# CrewAI configuration
# -----------------------------
resume_reviewer = Agent(
    role="이력서 평가관",
    goal="이력서를 객관적으로 평가하고 개선 방향을 제시",
    backstory="채용 실무 경험이 풍부한 리크루터 겸 피드백 코치",
    llm=llm,
    verbose=True,
)


def build_task(resume_text: str, job_requirements_text: str) -> Task:
    return Task(
        description=(
            "아래 이력서를 평가해라.\n"
            "요구 형식:\n"
            "1) 5~7줄 요약\n"
            "2) 강점 4~6개\n"
            "3) 약점/리스크 4~6개\n"
            "4) 개선 제안 6~10개 (즉시 적용 가능한 표현 포함)\n"
            "5) 직무 적합도 (0~100) + 점수 근거 3개\n"
            "6) ATS 관점 키워드 누락 5~10개\n\n"
            f"지원 직무: {INPUT['target_role']}\n\n"
            "직무요건:\n"
            f"{job_requirements_text}\n\n"
            "이력서 본문:\n"
            f"{resume_text}"
        ),
        expected_output="요구 형식에 맞춘 한국어 평가 리포트",
        agent=resume_reviewer,
    )


if __name__ == "__main__":
    resume_text = load_resume_text(INPUT["resume_path"])
    job_requirements_text = load_job_requirements(INPUT["job_requirements_path"])

    crew = Crew(
        agents=[resume_reviewer],
        tasks=[build_task(resume_text, job_requirements_text)],
        verbose=True,
    )

    result = crew.kickoff()
    print("\n\n===== RESULT =====\n")
    print(result)

