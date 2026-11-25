import logging
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Header
from pydantic import BaseModel, Field, field_validator
from app.core.config import get_settings
from app.rag.workflow import ask, extract_text_content
from app.rag.vectorstore import add_documents 
from app.services.ingestion import load_file, chunk_documents, SUPPORTED
from app.services.audit import write_audit

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")
settings = get_settings()

class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=3000)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        cleaned = value.strip()
        if len(cleaned) < 2:
            raise ValueError("Question must be at least 2 characters.")
        return cleaned


class Citation(BaseModel):
    title: str = ""
    url: str = ""
    type: str = ""


class ChatResponse(BaseModel):
    answer: str
    source_used: str
    trace: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    rewritten_query: str = ""


@router.get("/health")
def health():
    return {"status": "ok", "service": settings.app_name}


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    try:
        result = ask(payload.question)
        answer = extract_text_content(result.get("answer"))
        if not answer:
            answer = "I could not generate a text response. Please try again or contact the IT help desk."
        source_used = str(result.get("source_used") or "unknown")
        trace = [
            item if isinstance(item, str) else extract_text_content(item)
            for item in (result.get("trace") or [])
        ]
        citations: list[Citation] = []
        for c in result.get("citations") or []:
            if isinstance(c, dict):
                citations.append(
                    Citation(
                        title=extract_text_content(c.get("title")) or str(c.get("url") or "Source"),
                        url=str(c.get("url") or ""),
                        type=str(c.get("type") or ""),
                    )
                )
            elif isinstance(c, Citation):
                citations.append(c)
        rewritten_query = extract_text_content(result.get("current_query")) or payload.question

        write_audit(payload.question, source_used, trace)

        return ChatResponse(
            answer=answer,
            source_used=source_used,
            trace=trace,
            citations=citations,
            rewritten_query=rewritten_query,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Chat endpoint failed")
        raise HTTPException(status_code=500, detail=str(exc) or "Internal server error") from exc



@router.post("/ingest")
async def ingest(file: UploadFile = File(...), x_admin_key: str = Header(default="")):
    if x_admin_key != settings.admin_api_key:
        raise HTTPException(status_code=401, detail="Invalid admin key")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED:
        raise HTTPException(status_code=400, detail=f"Supported: {', '.join(sorted(SUPPORTED))}")
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / Path(file.filename).name
    dest.write_bytes(await file.read())
    docs = load_file(dest)
    chunks = chunk_documents(docs)
    ids = add_documents(chunks)
    return {"message": "Document indexed", "file": dest.name, "chunks": len(chunks), "ids_created": len(ids)}
