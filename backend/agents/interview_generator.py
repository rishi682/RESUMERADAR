import json
import re

from langchain_core.prompts import ChatPromptTemplate

from backend.agents.llm import get_llm
from backend.models.schemas import InterviewQuestion, InterviewQuestionSet, JobDescription, ParsedResume

_PROMPT = ChatPromptTemplate.from_template(
    """You are an experienced technical interviewer preparing questions for a candidate.

Resume skills: {resume_skills}
Resume raw text: {resume_text}

Job description: {job_description_text}
Required skills: {required_skills}

Generate exactly 5 interview questions tailored to this candidate and role.
Respond ONLY with valid JSON, no markdown, no code fences, in this exact structure:
{{"questions": [{{"question": "...", "category": "technical|behavioral|gap-probing", "reason": "..."}}]}}
"""
)


def _parse_llm_output(raw: str) -> InterviewQuestionSet:
    """Parse the LLM's JSON output into an InterviewQuestionSet, stripping any accidental markdown fences."""
    cleaned = re.sub(r"^```json\s*|\s*```$", "", raw.strip())
    data = json.loads(cleaned)
    questions = [InterviewQuestion(**q) for q in data.get("questions", [])]
    return InterviewQuestionSet(questions=questions)


def generate_interview_questions(
    parsed_resume: ParsedResume, job_description: JobDescription
) -> InterviewQuestionSet:
    """Generate LLM-based interview questions tailored to a resume/job description pairing."""
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