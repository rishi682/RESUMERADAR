from pydantic import BaseModel


class ExperienceEntry(BaseModel):
    """A single work experience entry extracted from a resume."""
    title: str
    company: str
    start_date: str | None
    end_date: str | None
    description: str
    
class EducationEntry(BaseModel):
    """A single education entry extracted from a resume."""
    degree: str
    institution: str
    year: str | None
    
class ParsedResume(BaseModel):
    """Structured data extracted from a raw resume text."""
    raw_text: str
    name: str | None
    email: str | None
    phone: str | None
    skills: list[str] = []
    experience: list[ExperienceEntry] = []
    education: list[EducationEntry] = []

class JobDescription(BaseModel):
    """Structured data extracted from a raw job description text."""
    raw_text: str
    title: str | None
    required_skills: list[str] = []
    responsibilities: list[str] = []
    
class MatchResult(BaseModel):
    """Scoring output comparing a parsed resume against a job description."""
    ats_score_percent: float  # keyword coverage only, 0-100
    match_score: float        # weighted keyword + semantic similarity, 0-100
    matched_keywords: list[str] = []
    missing_keywords: list[str] = []
    
class AnalyzeRequest(BaseModel):
    """Request payload for the resume/job analysis endpoint."""
    resume_text: str
    job_description_text: str
    
class AnalyzeResponse(BaseModel):
    """Response payload returned by the resume/job analysis endpoint."""
    parsed_resume: ParsedResume
    job_description: JobDescription
    match_result: MatchResult
    
class RecruiterFeedback(BaseModel):
    """LLM-generated recruiter-style feedback comparing a resume against a job description."""
    overall_impression: str
    strengths: list[str] = []
    concerns: list[str] = []
    verdict: str