


import json
from io import BytesIO

import httpx
import pytest
from fastapi.testclient import TestClient
from pypdf import PdfReader, PdfWriter

from app.main import app
from app.api import resume_analysis
from app.services.pdf_extraction import (
    EmptyPdfError,
    PdfExtractionError,
    extract_text_from_pdf,
)
from app.services.skill_identification import (
    OllamaInvalidOutputError,
    OllamaTimeoutError,
    OllamaUnavailableError,
    SkillIdentificationResult,
    identify_skills,
)

client = TestClient(app)


def make_pdf():
    """Create a valid PDF for endpoint tests."""
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)

    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def ollama_response(
    matched_required=None,
    matched_preferred=None,
    missing_required=None,
):
    """Build a mock Ollama response."""
    content = {
        "matched_required_skills": matched_required or [],
        "matched_preferred_skills": matched_preferred or [],
        "missing_required_skills": missing_required or [],
    }

    return httpx.Response(
        200,
        json={
            "message": {
                "content": json.dumps(content),
            }
        },
    )


def post_analysis(
    required=None,
    preferred=None,
    pdf_bytes=None,
    filename="resume.pdf",
    content_type="application/pdf",
):
    """Send a multipart request to the analysis endpoint."""
    form_data = {}

    if required:
        form_data["required_skills"] = required

    if preferred:
        form_data["preferred_skills"] = preferred

    return client.post(
        "/resume-analysis",
        data=form_data,
        files={
            "file": (
                filename,
                pdf_bytes if pdf_bytes is not None else make_pdf(),
                content_type,
            )
        },
    )


# -------------------------
# Health endpoint
# -------------------------

def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# -------------------------
# Match score tests
# -------------------------

def test_score_with_required_and_preferred_skills():
    score = resume_analysis.calculate_match_score(
        matched_required=["Python"],
        required=["Python", "SQL"],
        matched_preferred=["Docker"],
        preferred=["Docker", "Kubernetes"],
    )

    assert score == 50.0


def test_score_with_all_skills_matched():
    score = resume_analysis.calculate_match_score(
        matched_required=["Python", "SQL"],
        required=["Python", "SQL"],
        matched_preferred=["Docker"],
        preferred=["Docker"],
    )

    assert score == 100.0


def test_score_with_no_skills_matched():
    score = resume_analysis.calculate_match_score(
        matched_required=[],
        required=["Python", "SQL"],
        matched_preferred=[],
        preferred=["Docker"],
    )

    assert score == 0.0


def test_score_with_required_skills_only():
    score = resume_analysis.calculate_match_score(
        matched_required=["Python"],
        required=["Python", "SQL"],
        matched_preferred=[],
        preferred=[],
    )

    assert score == 50.0


def test_score_with_preferred_skills_only():
    score = resume_analysis.calculate_match_score(
        matched_required=[],
        required=[],
        matched_preferred=["Docker"],
        preferred=["Docker", "Kubernetes"],
    )

    assert score == 50.0


def test_score_with_no_skills():
    score = resume_analysis.calculate_match_score(
        matched_required=[],
        required=[],
        matched_preferred=[],
        preferred=[],
    )

    assert score == 0.0


# -------------------------
# Skill normalization tests
# -------------------------

def test_normalize_skills_removes_duplicates():
    result = resume_analysis.normalize_skills(
        ["Python", " SQL ", "python", "", "SQL"]
    )

    assert result == ["Python", "SQL"]


def test_normalize_skills_handles_none():
    assert resume_analysis.normalize_skills(None) == []


# -------------------------
# PDF extraction tests
# -------------------------

def test_extract_text_from_invalid_pdf():
    with pytest.raises(PdfExtractionError):
        extract_text_from_pdf(b"not a PDF")


def test_extract_text_from_blank_pdf():
    with pytest.raises(EmptyPdfError):
        extract_text_from_pdf(make_pdf())


# -------------------------
# API validation tests
# -------------------------

def test_rejects_non_pdf_filename():
    response = post_analysis(
        required=["Python"],
        filename="resume.txt",
    )

    assert response.status_code == 400


def test_rejects_non_pdf_content():
    response = post_analysis(
        required=["Python"],
        pdf_bytes=b"this is not a PDF",
    )

    assert response.status_code == 400



def test_rejects_missing_skills(monkeypatch):
    monkeypatch.setattr(
        "app.api.resume_analysis.extract_text_from_pdf",
        lambda data: "Sample resume text",
    )

    response = post_analysis()

    assert response.status_code == 422
    assert "skill" in response.json()["detail"].lower()
    assert "provide at least one" in response.json()["detail"].lower()


def test_rejects_blank_pdf():
    response = post_analysis(
        required=["Python"],
        pdf_bytes=make_pdf(),
    )

    assert response.status_code == 422


# -------------------------
# Skill identification tests
# -------------------------

def test_skill_result_structure():
    result = SkillIdentificationResult(
        matched_required_skills=["Python"],
        matched_preferred_skills=["Docker"],
        missing_required_skills=["SQL"],
    )

    assert result.matched_required_skills == ["Python"]
    assert result.matched_preferred_skills == ["Docker"]
    assert result.missing_required_skills == ["SQL"]


def test_rejects_ai_skill_not_in_job_list(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "message": {
                    "content": json.dumps({
                        "matched_required_skills": ["Java"],
                        "matched_preferred_skills": [],
                        "missing_required_skills": [],
                    })
                }
            }

    monkeypatch.setattr(
        "app.services.skill_identification._post_chat",
        lambda payload: FakeResponse(),
    )

    with pytest.raises(OllamaInvalidOutputError):
        identify_skills(
            "Resume mentions Python",
            ["Python"],
            [],
        )


def test_ai_connection_failure(monkeypatch):
    def raise_connection_error(payload):
        raise httpx.ConnectError("Connection failed")

    monkeypatch.setattr(
        "app.services.skill_identification._post_chat",
        raise_connection_error,
    )

    with pytest.raises(OllamaUnavailableError):
        identify_skills("Resume text", ["Python"], [])


def test_ai_timeout(monkeypatch):
    def raise_timeout(payload):
        raise httpx.TimeoutException("Timed out")

    monkeypatch.setattr(
        "app.services.skill_identification._post_chat",
        raise_timeout,
    )

    with pytest.raises(OllamaTimeoutError):
        identify_skills("Resume text", ["Python"], [])


def test_malformed_ai_json(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "message": {
                    "content": "not valid JSON",
                }
            }

    monkeypatch.setattr(
        "app.services.skill_identification._post_chat",
        lambda payload: FakeResponse(),
    )

    with pytest.raises(OllamaInvalidOutputError):
        identify_skills("Resume text", ["Python"], [])


# -------------------------
# Successful API analysis
# -------------------------

def test_valid_ai_result(monkeypatch):
    monkeypatch.setattr(
        "app.api.resume_analysis.extract_text_from_pdf",
        lambda data: "Python and Docker experience",
    )

    monkeypatch.setattr(
        "app.api.resume_analysis.identify_skills",
        lambda text, required, preferred: SkillIdentificationResult(
            matched_required_skills=["Python"],
            matched_preferred_skills=["Docker"],
            missing_required_skills=["SQL"],
        ),
    )

    response = post_analysis(
        required=["Python", "SQL"],
        preferred=["Docker", "Kubernetes"],
    )

    assert response.status_code == 200

    body = response.json()

    assert body["matched_required_skills"] == ["Python"]
    assert body["matched_preferred_skills"] == ["Docker"]
    assert body["missing_required_skills"] == ["SQL"]
    assert body["match_score"] == 50.0


def test_api_returns_503_when_ai_times_out(monkeypatch):
    monkeypatch.setattr(
        "app.api.resume_analysis.extract_text_from_pdf",
        lambda data: "Python experience",
    )

    def raise_timeout(*args, **kwargs):
        raise OllamaTimeoutError()

    monkeypatch.setattr(
        "app.api.resume_analysis.identify_skills",
        raise_timeout,
    )

    response = post_analysis(required=["Python"])

    assert response.status_code == 503


def test_api_returns_503_when_ai_unavailable(monkeypatch):
    monkeypatch.setattr(
        "app.api.resume_analysis.extract_text_from_pdf",
        lambda data: "Python experience",
    )

    def raise_unavailable(*args, **kwargs):
        raise OllamaUnavailableError()

    monkeypatch.setattr(
        "app.api.resume_analysis.identify_skills",
        raise_unavailable,
    )

    response = post_analysis(required=["Python"])

    assert response.status_code == 503


def test_api_returns_502_for_invalid_ai_output(monkeypatch):
    monkeypatch.setattr(
        "app.api.resume_analysis.extract_text_from_pdf",
        lambda data: "Python experience",
    )

    def raise_invalid_output(*args, **kwargs):
        raise OllamaInvalidOutputError()

    monkeypatch.setattr(
        "app.api.resume_analysis.identify_skills",
        raise_invalid_output,
    )

    response = post_analysis(required=["Python"])

    assert response.status_code == 502
