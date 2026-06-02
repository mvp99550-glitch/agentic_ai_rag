"""Pydantic request/response schemas — shared between FastAPI and Streamlit."""

import re
from pydantic import BaseModel, field_validator, model_validator
from typing import Optional

_SAFE_ID = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")


class QueryRequest(BaseModel):
    question:   str
    tenant_id:  str
    user_id:    str
    session_id: str
    top_k:      int = 5

    @field_validator("question")
    @classmethod
    def question_clean(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Question cannot be empty.")
        if len(v) > 500:
            raise ValueError("Question must be 500 characters or fewer.")
        return v

    @field_validator("tenant_id", "user_id", "session_id")
    @classmethod
    def id_safe(cls, v: str) -> str:
        v = v.strip()
        if not _SAFE_ID.match(v):
            raise ValueError("ID must be alphanumeric (a-z, 0-9, _, -) and max 64 chars.")
        return v

    @field_validator("top_k")
    @classmethod
    def top_k_range(cls, v: int) -> int:
        if not (1 <= v <= 10):
            raise ValueError("top_k must be between 1 and 10.")
        return v


class Citation(BaseModel):
    number:     int
    page:       Optional[int]
    section:    Optional[str]
    source:     Optional[str]   # "text" | "table"
    collection: str


class RetrievedChunk(BaseModel):
    citation_number: int
    text:       str
    score:      float
    page:       Optional[int]
    section:    Optional[str]
    source:     Optional[str]
    collection: str


class QueryResponse(BaseModel):
    question:   str
    tenant_id:  str
    user_id:    str
    session_id: str
    chunks:     list[RetrievedChunk]
    citations:  list[Citation]
    answer:     Optional[str] = None   # None until LLM is wired in
    llm_ready:  bool = False
