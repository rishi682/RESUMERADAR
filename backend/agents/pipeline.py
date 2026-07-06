from typing import TypedDict

from langgraph.graph import END, StateGraph

from backend.database.hybrid_retriever import hybrid_search
from backend.models.schemas import JobDescription, MatchResult, ParsedResume
from backend.nlp.extractor import parse_resume


class PipelineState(TypedDict):
    """Shared state passed between LangGraph nodes."""
    resume_text: str
    job_description_text: str
    parsed_resume: ParsedResume | None
    job_description: JobDescription | None
    match_result: MatchResult | None


def _parse_resume_node(state: PipelineState) -> PipelineState:
    """Parse the raw resume text into a ParsedResume."""
    parsed = parse_resume(state["resume_text"])
    return {**state, "parsed_resume": parsed}


def _parse_job_description_node(state: PipelineState) -> PipelineState:
    """Parse the raw job description text into a JobDescription."""
    from backend.nlp.extractor import extract_skills

    text = state["job_description_text"]
    job_description = JobDescription(
        raw_text=text,
        title=None,
        required_skills=extract_skills(text),
        responsibilities=[],
    )
    return {**state, "job_description": job_description}


def _score_node(state: PipelineState) -> PipelineState:
    """Compute ats_score_percent and match_score by comparing resume skills against required skills."""
    resume_skills = {s.lower() for s in state["parsed_resume"].skills}
    required_skills = {s.lower() for s in state["job_description"].required_skills}

    if not required_skills:
        ats_score_percent = 0.0
        matched = []
        missing = []
    else:
        matched = sorted(resume_skills & required_skills)
        missing = sorted(required_skills - resume_skills)
        ats_score_percent = round(100 * len(matched) / len(required_skills), 2)

    candidate_texts = [state["resume_text"], state["job_description_text"]]
    candidate_ids = ["resume", "job_description"]
    fused = hybrid_search(
        query_text=state["job_description_text"],
        corpus=candidate_texts,
        doc_ids=candidate_ids,
        n_results=2,
    )
    resume_score = next((score for doc_id, score in fused if doc_id == "resume"), 0.0)
    match_score = round(min(resume_score * 1000, 100.0), 2)

    match_result = MatchResult(
        ats_score_percent=ats_score_percent,
        match_score=match_score,
        matched_keywords=matched,
        missing_keywords=missing,
    )
    return {**state, "match_result": match_result}


def build_pipeline():
    """Build and compile the LangGraph StateGraph for the resume analysis pipeline."""
    graph = StateGraph(PipelineState)

    graph.add_node("parse_resume", _parse_resume_node)
    graph.add_node("parse_job_description", _parse_job_description_node)
    graph.add_node("score", _score_node)

    graph.set_entry_point("parse_resume")
    graph.add_edge("parse_resume", "parse_job_description")
    graph.add_edge("parse_job_description", "score")
    graph.add_edge("score", END)

    return graph.compile()