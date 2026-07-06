from fastapi import APIRouter

from backend.agents.interview_generator import generate_interview_questions
from backend.agents.pipeline import build_pipeline
from backend.agents.recruiter_simulator import simulate_recruiter_review
from backend.models.schemas import AnalyzeRequest, AnalyzeResponse, InterviewQuestionSet, RecruiterFeedback
from backend.nlp.extractor import extract_skills, parse_resume
from backend.models.schemas import JobDescription

router = APIRouter()

_pipeline = build_pipeline()


@router.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """Run the resume/job-description analysis pipeline and return structured results."""
    result = _pipeline.invoke(
        {
            "resume_text": request.resume_text,
            "job_description_text": request.job_description_text,
            "parsed_resume": None,
            "job_description": None,
            "match_result": None,
        }
    )
    return AnalyzeResponse(
        parsed_resume=result["parsed_resume"],
        job_description=result["job_description"],
        match_result=result["match_result"],
    )


@router.post("/api/recruiter-simulation", response_model=RecruiterFeedback)
async def recruiter_simulation(request: AnalyzeRequest) -> RecruiterFeedback:
    """Generate LLM-based recruiter feedback for a resume/job description pair."""
    parsed_resume = parse_resume(request.resume_text)
    job_description = JobDescription(
        raw_text=request.job_description_text,
        title=None,
        required_skills=extract_skills(request.job_description_text),
        responsibilities=[],
    )
    return simulate_recruiter_review(parsed_resume, job_description)


@router.post("/api/interview-questions", response_model=InterviewQuestionSet)
async def interview_questions(request: AnalyzeRequest) -> InterviewQuestionSet:
    """Generate LLM-based interview questions for a resume/job description pair."""
    parsed_resume = parse_resume(request.resume_text)
    job_description = JobDescription(
        raw_text=request.job_description_text,
        title=None,
        required_skills=extract_skills(request.job_description_text),
        responsibilities=[],
    )
    return generate_interview_questions(parsed_resume, job_description)