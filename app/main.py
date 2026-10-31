"""FastAPI application entry point."""
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import router
from app.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)

app = FastAPI(
    title="Clinical RAG Prototype",
    description=(
        "An Evidence-Based Clinical AI Assistant for Healthcare Professionals "
        "Using Medical Large Language Models and Retrieval-Augmented Generation. "
        "Academic prototype only — not clinically validated."
    ),
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow Streamlit UI (served on a different port) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router)
