"""SQLite database session management using SQLModel."""
from pathlib import Path
from typing import Generator

from sqlmodel import Session, SQLModel, create_engine

from app.config import settings

_DB_PATH = settings.data_dir / "app.db"
_DB_URL = f"sqlite:///{_DB_PATH}"

# check_same_thread=False is required for SQLite + FastAPI (multiple request threads)
_engine = create_engine(
    _DB_URL,
    echo=False,
    connect_args={"check_same_thread": False},
)


def get_engine():
    """Return the application database engine."""
    return _engine


def create_db_and_tables(engine=None) -> None:
    """Create all SQLModel table schemas in the database (idempotent)."""
    target = engine or _engine
    # Import models to ensure all SQLModel table classes are registered
    import app.models  # noqa: F401
    SQLModel.metadata.create_all(target)


def get_session(engine=None) -> Generator[Session, None, None]:
    """FastAPI dependency that yields a database session."""
    target = engine or _engine
    with Session(target) as session:
        yield session
