
from pydantic import BaseModel, Field


class ResumeAnalysisResponse(BaseModel):
    matched_required_skills: list[str] = Field(default_factory=list)
    matched_preferred_skills: list[str] = Field(default_factory=list)
    missing_required_skills: list[str] = Field(default_factory=list)
    match_score: float = Field(ge=0, le=100)
