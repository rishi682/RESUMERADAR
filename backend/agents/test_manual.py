from backend.agents.pipeline import build_pipeline

pipeline = build_pipeline()

resume_text = """RISHI SINGH GURJAR
Summary
AI Systems Engineer focused on building reliable and scalable AI applications using LLMs and multi-agent architectures.
Technical Skills
Programming: Python, C++
Full-Stack Development: React.js, Node.js, Express.js, MongoDB
Education
B.Tech in Computer Science Engineering (AI/ML) Expected 2027
Bennett University, Greater Noida"""

job_description_text = """We are looking for a backend engineer with strong Python and FastAPI experience.
Experience with LangGraph, ChromaDB, and hybrid search systems is a big plus.
Familiarity with Docker and Kubernetes is required."""

result = pipeline.invoke({
    "resume_text": resume_text,
    "job_description_text": job_description_text,
    "parsed_resume": None,
    "job_description": None,
    "match_result": None,
})

print(result["match_result"])