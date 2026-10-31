"""Application configuration and constants."""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Storage paths
    root_dir: Path = ROOT_DIR
    data_dir: Path = ROOT_DIR / "data"
    raw_dir: Path = ROOT_DIR / "data" / "raw"
    manifest_dir: Path = ROOT_DIR / "data" / "manifest"
    passages_dir: Path = ROOT_DIR / "data" / "passages"
    benchmark_dir: Path = ROOT_DIR / "data" / "benchmark"
    chroma_dir: Path = ROOT_DIR / "chroma_db"
    aliases_path: Path = ROOT_DIR / "data" / "manifest" / "aliases.json"
    documents_manifest_path: Path = ROOT_DIR / "data" / "manifest" / "documents.json"

    # ChromaDB settings
    chroma_collection_name: str = "drug_passages"
    chroma_distance_fn: str = "cosine"
    retrieval_top_k: int = 5

    # Groq & Model settings
    groq_model: str = "openai/gpt-oss-120b"
    groq_api_key: str = ""
    groq_timeout_seconds: float = 45.0
    groq_temperature: float = 0.0
    groq_max_tokens: int = 2048
    groq_reasoning_effort: str = "low"
    prompt_version: str = "v1"

    # Fixed project settings
    fixed_project_title: str = (
        "An Evidence-Based Clinical AI Assistant for Healthcare Professionals "
        "Using Medical Large Language Models and Retrieval-Augmented Generation"
    )
    fixed_model_sentence: str = (
        "The prototype uses the general-purpose openai/gpt-oss-120b model "
        "through Groq within a medical-information RAG workflow."
    )


settings = Settings()
