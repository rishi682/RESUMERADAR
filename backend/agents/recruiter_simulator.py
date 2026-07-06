from langchain_core.prompts import ChatPromptTemplate

from backend.agents.llm import get_llm
from backend.models.schemas import JobDescription, ParsedResume, RecruiterFeedback

_PROMPT = ChatPromptTemplate.from_template(
    """You are an experienced technical recruiter reviewing a candidate for a specific role.

Resume skills: {resume_skills}
Resume raw text: {resume_text}

Job description: {job_description_text}
Required skills: {required_skills}

Respond ONLY in this exact format, nothing else:
OVERALL: <one or two sentence overall impression>
STRENGTHS: <comma-separated list of strengths relevant to this role>
CONCERNS: <comma-separated list of concerns or gaps>
VERDICT: <one short sentence: would you move this candidate forward or not, and why>
"""
)


def _parse_llm_output(raw: str) -> RecruiterFeedback:
    """Parse the LLM's structured text output into a RecruiterFeedback object."""
    lines = {}
    for line in raw.strip().splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            lines[key.strip().upper()] = value.strip()

    strengths = [s.strip() for s in lines.get("STRENGTHS", "").split(",") if s.strip()]
    concerns = [c.strip() for c in lines.get("CONCERNS", "").split(",") if c.strip()]

    return RecruiterFeedback(
        overall_impression=lines.get("OVERALL", ""),
        strengths=strengths,
        concerns=concerns,
        verdict=lines.get("VERDICT", ""),
    )


def simulate_recruiter_review(
    parsed_resume: ParsedResume, job_description: JobDescription
) -> RecruiterFeedback:
    """Generate LLM-based recruiter feedback comparing a resume against a job description."""
    llm = get_llm()
    chain = _PROMPT | llm
    response = chain.invoke(
        {
            "resume_skills": ", ".join(parsed_resume.skills),
            "resume_text": parsed_resume.raw_text,
            "job_description_text": job_description.raw_text,
            "required_skills": ", ".join(job_description.required_skills),
        }
    )
    return _parse_llm_output(response.content)