from fastapi import FastAPI

from app.api.resume_analysis import router as resume_analysis_router

app = FastAPI(title="CS455 Group 10 API")
app.include_router(resume_analysis_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

