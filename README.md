## KAN-18: Resume–Job Analysis

### Overview

KAN-18 analyzes a candidate's resume against the required and preferred skills for a job. It extracts text from a PDF resume, uses a locally hosted Ollama model to identify matching skills, validates the model output, and calculates a deterministic match score.

### Technology Stack

- Python
- FastAPI
- Pydantic
- pypdf
- Ollama (`qwen3:4b`)
- Pytest

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Checks whether the backend is running. |
| POST | `/resume-analysis` | Analyzes a resume against job skills. |

### Request Parameters

The `POST /resume-analysis` endpoint accepts multipart form data:

- `file`: Resume PDF, maximum size 5 MB.
- `required_skills`: Required job skills.
- `preferred_skills`: Optional preferred job skills.

At least one required or preferred skill must be provided. Skills may be supplied as comma-separated values.

### Response Fields

- `matched_required_skills`: Required skills supported by the resume.
- `matched_preferred_skills`: Preferred skills supported by the resume.
- `missing_required_skills`: Required skills not matched.
- `match_score`: Calculated score from 0 to 100.

### Match Score Calculation

When both required and preferred skills are provided:

`Match Score = 70 × Required Match Ratio + 30 × Preferred Match Ratio`

Each match ratio is the number of matched skills divided by the total skills in that category.

If only one category is provided, its match ratio is normalized to a score out of 100. If neither category contains skills, the score calculation returns 0; however, the API rejects requests with no skills.

### Error Responses

| Status | Meaning |
|---|---|
| 400 | Invalid file type or PDF filename. |
| 413 | Resume exceeds the 5 MB limit. |
| 422 | No extractable PDF text or no skills supplied. |
| 502 | Invalid skill-identification output from Ollama. |
| 503 | Ollama unavailable or request timed out. |

### Local Setup

1. Install Python and Ollama.
2. Download the model:

   `ollama pull qwen3:4b`

3. Install backend dependencies from the `backend` directory:

   `pip install -r requirements.txt`

4. Start the API:

   `uvicorn app.main:app --reload`

5. Open `http://127.0.0.1:8000/docs` to test the endpoints interactively.

Ensure Ollama is running locally before testing resume analysis.

### Running Tests

From the `backend` directory, run:

` .venv/bin/python -m pytest tests/ -v `

The tests cover score calculations, skill normalization, PDF validation, AI response validation, and API error handling.