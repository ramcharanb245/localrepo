
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.schemas.resume import ResumeAnalysisResponse
from app.services.pdf_extraction import (
    EmptyPdfError,
    PdfExtractionError,
    extract_text_from_pdf,
)
from app.services.skill_identification import (
    OllamaInvalidOutputError,
    OllamaTimeoutError,
    OllamaUnavailableError,
    identify_skills,
)

router = APIRouter(tags=["resume-analysis"])

MAX_RESUME_BYTES = 5 * 1024 * 1024
READ_CHUNK_BYTES = 64 * 1024
PDF_MAGIC = b"%PDF"


def calculate_match_score(
    matched_required: list[str],
    required: list[str],
    matched_preferred: list[str],
    preferred: list[str],
) -> float:
    """Calculate the deterministic 70/30 resume-job match score."""
    if not required and not preferred:
        return 0.0

    required_ratio = (
        len(matched_required) / len(required) if required else None
    )
    preferred_ratio = (
        len(matched_preferred) / len(preferred) if preferred else None
    )

    if required_ratio is None:
        return round(preferred_ratio * 100, 2)

    if preferred_ratio is None:
        return round(required_ratio * 100, 2)

    return round(
        70 * required_ratio + 30 * preferred_ratio,
        2,
    )


async def read_pdf_capped(file: UploadFile) -> bytes:
    chunks = []
    total = 0

    while True:
        chunk = await file.read(READ_CHUNK_BYTES)
        if not chunk:
            break

        total += len(chunk)
        if total > MAX_RESUME_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail="Resume file exceeds the 5 MB limit.",
            )

        chunks.append(chunk)

    return b"".join(chunks)



def normalize_skills(skills: list[str] | None) -> list[str]:
    result = []
    seen = set()

    for item in skills or []:
        # Accept comma-separated skills as well as separate fields.
        for part in item.split(","):
            name = part.strip()
            key = name.casefold()

            if name and key not in seen:
                result.append(name)
                seen.add(key)

    return result



@router.post(
    "/resume-analysis",
    response_model=ResumeAnalysisResponse,
)
async def analyze_resume(
    file: UploadFile = File(..., description="Resume PDF"),
    required_skills: list[str] | None = Form(default=None),
    preferred_skills: list[str] | None = Form(default=None),
) -> ResumeAnalysisResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF resume files are accepted.",
        )

    data = await read_pdf_capped(file)

    if not data.startswith(PDF_MAGIC):
        raise HTTPException(
            status_code=400,
            detail="Only PDF resume files are accepted.",
        )

    try:
        text = extract_text_from_pdf(data)
    except EmptyPdfError:
        raise HTTPException(
            status_code=422,
            detail="PDF contains no extractable text.",
        ) from None
    except PdfExtractionError:
        raise HTTPException(
            status_code=500,
            detail="Failed to extract text from PDF.",
        ) from None

    required = normalize_skills(required_skills)
    preferred = normalize_skills(preferred_skills)

    if not required and not preferred:
        raise HTTPException(
            status_code=422,
            detail="Provide at least one required or preferred skill.",
        )

    try:
        result = identify_skills(text, required, preferred)
    except OllamaTimeoutError:
        raise HTTPException(
            status_code=503,
            detail="Skill identification timed out.",
        ) from None
    except OllamaUnavailableError:
        raise HTTPException(
            status_code=503,
            detail="Skill identification service is unavailable.",
        ) from None
    except OllamaInvalidOutputError:
        raise HTTPException(
            status_code=502,
            detail="Skill identification returned invalid output.",
        ) from None

    score = calculate_match_score(
        result.matched_required_skills,
        required,
        result.matched_preferred_skills,
        preferred,
    )

    return ResumeAnalysisResponse(
        matched_required_skills=result.matched_required_skills,
        matched_preferred_skills=result.matched_preferred_skills,
        missing_required_skills=result.missing_required_skills,
        match_score=score,
    )
