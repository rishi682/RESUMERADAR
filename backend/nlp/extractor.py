import json
import re
from pathlib import Path

import spacy
from skillNer.general_params import SKILL_DB
from skillNer.skill_extractor_class import SkillExtractor
from spacy.matcher import PhraseMatcher

from backend.models.schemas import EducationEntry, ExperienceEntry, ParsedResume

_nlp = spacy.load("en_core_web_lg")
_skill_extractor = SkillExtractor(_nlp, SKILL_DB, PhraseMatcher)

_SUPPLEMENTAL_SKILLS_PATH = Path(__file__).parent / "supplemental_skills.json"
with open(_SUPPLEMENTAL_SKILLS_PATH, encoding="utf-8") as f:
    _SUPPLEMENTAL_SKILLS = json.load(f)["skills"]

EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_PATTERN = re.compile(r"(\+?\d{1,3}[-.\s]?)?\(?\d{3,5}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}")

SECTION_HEADERS = {
    "experience": ["experience", "work experience", "employment history"],
    "education": ["education", "academic background"],
    "projects": ["projects"],
}


def extract_email(text: str) -> str | None:
    """Return the first email address found in the text, if any."""
    match = EMAIL_PATTERN.search(text)
    return match.group(0) if match else None


def extract_phone(text: str) -> str | None:
    """Return the first phone number found in the text, if any."""
    match = PHONE_PATTERN.search(text)
    return match.group(0) if match else None


def extract_name(text: str) -> str | None:
    """Return the first detected PERSON entity, assumed to be the candidate's name."""
    doc = _nlp(text[:500])
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return None


def extract_skills(text: str) -> list[str]:
    """Return a deduplicated list of skills detected via SkillNER plus a supplemental list for niche AI/ML terms SkillNER misses."""
    annotations = _skill_extractor.annotate(text)
    found = set()
    for match in annotations["results"]["full_matches"]:
        found.add(match["doc_node_value"])
    for match in annotations["results"]["ngram_scored"]:
        found.add(match["doc_node_value"])

    text_lower = text.lower()
    for skill in _SUPPLEMENTAL_SKILLS:
        if skill.lower() in text_lower:
            found.add(skill)

    return sorted(found)


def _split_sections(text: str) -> dict[str, str]:
    """Split resume text into sections keyed by section name, based on known headers."""
    lines = text.splitlines()
    header_to_section = {}
    for section, headers in SECTION_HEADERS.items():
        for header in headers:
            header_to_section[header] = section

    sections: dict[str, list[str]] = {}
    current_section = None
    for line in lines:
        stripped_lower = line.strip().lower()
        if stripped_lower in header_to_section:
            current_section = header_to_section[stripped_lower]
            sections[current_section] = []
        elif current_section is not None:
            sections[current_section].append(line)

    return {name: "\n".join(content).strip() for name, content in sections.items()}


def extract_education(text: str) -> list[EducationEntry]:
    """Return education entries parsed from the education section, if present."""
    sections = _split_sections(text)
    education_text = sections.get("education", "")
    if not education_text:
        return []

    entries = []
    lines = [line.strip() for line in education_text.splitlines() if line.strip()]
    for i in range(0, len(lines) - 1, 2):
        degree_line = lines[i]
        institution_line = lines[i + 1]

        year_match = re.search(r"(19|20)\d{2}", degree_line)
        year = year_match.group(0) if year_match else None
        degree = re.sub(r"Expected\s+\d{4}|\b(19|20)\d{2}\b", "", degree_line).strip()

        entries.append(
            EducationEntry(degree=degree, institution=institution_line, year=year)
        )
    return entries


def extract_experience(text: str) -> list[ExperienceEntry]:
    """Return experience entries parsed from the experience section, if present. Returns an empty list if no experience section exists (e.g. student resumes with only a Projects section)."""
    sections = _split_sections(text)
    experience_text = sections.get("experience", "")
    if not experience_text:
        return []
    return []


def parse_resume(raw_text: str) -> ParsedResume:
    """Run the full extraction pipeline on raw resume text and return a ParsedResume."""
    return ParsedResume(
        raw_text=raw_text,
        name=extract_name(raw_text),
        email=extract_email(raw_text),
        phone=extract_phone(raw_text),
        skills=extract_skills(raw_text),
        experience=extract_experience(raw_text),
        education=extract_education(raw_text),
    )