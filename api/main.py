"""
HyeAero.AI — Aircraft Research & Valuation Consultant API

Real-time AI assistant for aircraft research, market comparison, predictive pricing,
and resale advisory using Hye Aero's proprietary data (Controller, AircraftExchange, FAA, Internal DB).
"""

import os
import sys
from pathlib import Path

# Backend root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Any

from config.config_loader import Config  # Config.from_env() without validate for flexible API
from database.postgres_client import PostgresClient
from rag.embedding_service import EmbeddingService
from rag.query_service import RAGQueryService
from vector_store.pinecone_client import PineconeClient
from services.market_comparison import run_comparison
from services.price_estimate import estimate_value, estimate_value_hybrid

# --- Pydantic models ---
class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=0)

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    history: Optional[List[ChatMessage]] = None  # prior conversation for context

class ChatResponse(BaseModel):
    answer: str
    sources: List[Any] = []
    data_used: Optional[dict] = None  # e.g. {"aircraft listing": 3, "aircraft sale": 2}
    error: Optional[str] = None

class MarketComparisonRequest(BaseModel):
    models: List[str] = Field(..., min_length=1)
    region: Optional[str] = None
    max_hours: Optional[float] = None
    min_year: Optional[int] = None
    max_year: Optional[int] = None
    limit: int = Field(50, ge=1, le=200)

class PriceEstimateRequest(BaseModel):
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    year: Optional[int] = None
    flight_hours: Optional[float] = None
    flight_cycles: Optional[int] = None
    region: Optional[str] = None

class ResaleAdvisoryRequest(BaseModel):
    query: Optional[str] = None
    listing_id: Optional[str] = None
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    year: Optional[int] = None

# --- App ---
app = FastAPI(
    title="HyeAero.AI API",
    description="Aircraft Research & Valuation Consultant — market comparison, pricing, resale advisory, RAG chat",
    version="1.0.0",
)

# CORS: set CORS_ORIGINS on Render to your frontend URL (e.g. https://your-app.onrender.com)
_default_cors = "http://localhost:3000,http://127.0.0.1:3000,http://88.99.198.243:3000,http://88.99.198.243"
_cors_raw = os.getenv("CORS_ORIGINS", _default_cors).strip().split(",")
_allow_origins = [o.strip() for o in _cors_raw if o.strip()]
if not _allow_origins:
    _allow_origins = _default_cors.strip().split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded dependencies
_config: Optional[Config] = None
_db: Optional[PostgresClient] = None
_rag: Optional[RAGQueryService] = None

def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config.from_env()  # Do not call .validate() so API can run with partial config
    return _config

def get_db() -> PostgresClient:
    global _db
    if _db is None:
        c = get_config()
        if not c.postgres_connection_string:
            raise HTTPException(status_code=503, detail="PostgreSQL not configured")
        _db = PostgresClient(c.postgres_connection_string)
    return _db

def get_rag() -> RAGQueryService:
    global _rag
    if _rag is None:
        c = get_config()
        if not all([c.openai_api_key, c.pinecone_api_key, c.postgres_connection_string]):
            raise HTTPException(status_code=503, detail="RAG not configured (OpenAI, Pinecone, Postgres)")
        emb = EmbeddingService(
            api_key=c.openai_api_key,
            model=c.openai_embedding_model,
            dimension=c.openai_embedding_dimension,
        )
        pc = PineconeClient(
            api_key=c.pinecone_api_key,
            index_name=c.pinecone_index_name,
            dimension=c.pinecone_dimension,
            metric=c.pinecone_metric,
            host=c.pinecone_host,
        )
        if not pc.connect():
            raise HTTPException(status_code=503, detail="Pinecone connection failed")
        _rag = RAGQueryService(
            embedding_service=emb,
            pinecone_client=pc,
            postgres_client=PostgresClient(c.postgres_connection_string),
            openai_api_key=c.openai_api_key,
            chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        )
    return _rag


def get_embedding_and_pinecone():
    """Return (embedding_service, pinecone_client) when configured and connected; else (None, None)."""
    c = get_config()
    if not c.openai_api_key or not c.pinecone_api_key:
        return None, None
    try:
        emb = EmbeddingService(
            api_key=c.openai_api_key,
            model=c.openai_embedding_model,
            dimension=c.openai_embedding_dimension,
        )
        pc = PineconeClient(
            api_key=c.pinecone_api_key,
            index_name=c.pinecone_index_name,
            dimension=c.pinecone_dimension,
            metric=c.pinecone_metric,
            host=c.pinecone_host,
        )
        if not pc.connect():
            return None, None
        return emb, pc
    except Exception:
        return None, None

@app.get("/")
def root():
    return {"service": "HyeAero.AI", "version": "1.0.0", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/aircraft-models")
def aircraft_models():
    """Return distinct aircraft models (manufacturer + model) that have at least one listing.
    This ensures the Market Comparison dropdown only shows models that can return comparison results."""
    try:
        db = get_db()
        rows = db.execute_query(
            """
            SELECT DISTINCT a.manufacturer, a.model
            FROM aircraft_listings l
            INNER JOIN aircraft a ON l.aircraft_id = a.id
            WHERE ((a.manufacturer IS NOT NULL AND TRIM(a.manufacturer) != '')
               OR (a.model IS NOT NULL AND TRIM(a.model) != ''))
            ORDER BY a.manufacturer, a.model
            """
        )
        seen = set()
        models = []
        for r in rows:
            man = (r.get("manufacturer") or "").strip()
            mod = (r.get("model") or "").strip()
            label = f"{man} {mod}".strip()
            if label and label not in seen:
                seen.add(label)
                models.append(label)
        return {"models": models}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/price-estimate-models")
def price_estimate_models():
    """Return distinct aircraft models from aircraft_sales that have at least one sale.
    Use this list for the Price Estimator dropdown so selected models always have comparable sales."""
    try:
        db = get_db()
        rows = db.execute_query(
            """
            SELECT DISTINCT manufacturer, model
            FROM aircraft_sales
            WHERE sold_price IS NOT NULL AND sold_price > 0
              AND (COALESCE(manufacturer,'') != '' OR COALESCE(model,'') != '')
            ORDER BY manufacturer, model
            """
        )
        seen = set()
        models = []
        for r in rows:
            man = (r.get("manufacturer") or "").strip()
            mod = (r.get("model") or "").strip()
            label = f"{man} {mod}".strip()
            if label and label not in seen:
                seen.add(label)
                models.append(label)
        sample = None
        test_payloads = []
        if models:
            sample = {"model": models[0], "region": "Global"}
            # First 5 models as test values (exact strings from DB — will return results)
            for m in models[:5]:
                test_payloads.append({"model": m, "region": "Global"})
        return {"models": models, "sample_request": sample, "test_payloads": test_payloads}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/rag/answer", response_model=ChatResponse)
def rag_answer(req: ChatRequest):
    """Ask the Consultant: natural language over Hye Aero's sale history + market feed (RAG)."""
    try:
        rag = get_rag()
        history_dicts = [{"role": m.role, "content": m.content} for m in (req.history or [])]
        out = rag.answer(query=req.query.strip(), top_k=20, history=history_dicts if history_dicts else None)
        if out.get("error"):
            raise HTTPException(status_code=500, detail=out["error"])
        return ChatResponse(
            answer=out["answer"],
            sources=out.get("sources", []),
            data_used=out.get("data_used"),
            error=out.get("error"),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/market-comparison")
def market_comparison(req: MarketComparisonRequest):
    """Dynamic market comparison: compare aircraft by model, age, hours, region (Controller, AircraftExchange, FAA, Internal DB)."""
    try:
        db = get_db()
        result = run_comparison(
            db=db,
            models=req.models,
            region=req.region,
            max_hours=req.max_hours,
            min_year=req.min_year,
            max_year=req.max_year,
            limit=req.limit,
        )
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/price-estimate")
def price_estimate(req: PriceEstimateRequest):
    """Predictive pricing: vector search first (Pinecone), then full details from PostgreSQL;
    if no vector results, fall back to PostgreSQL keyword search."""
    try:
        db = get_db()
        emb, pc = get_embedding_and_pinecone()
        result = estimate_value_hybrid(
            db=db,
            embedding_service=emb,
            pinecone_client=pc,
            manufacturer=req.manufacturer,
            model=req.model,
            year=req.year,
            flight_hours=req.flight_hours,
            flight_cycles=req.flight_cycles,
            region=req.region,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/resale-advisory")
def resale_advisory(req: ResaleAdvisoryRequest):
    """Resale advisory: plain-English guidance (e.g. undervalued by X% given comparables). Uses RAG when query provided."""
    try:
        if req.query:
            rag = get_rag()
            out = rag.answer(
                query=f"Resale advisory: {req.query}. Provide plain-English guidance on whether this aircraft is over/undervalued and why, given comparable sales and market data. Use plain text and bullet points (-) only; do not use markdown headers (#) or double asterisks (**). You may use professional symbols (e.g. •, ✓, →) where appropriate.",
                top_k=8,
            )
            if out.get("error"):
                raise HTTPException(status_code=500, detail=out["error"])
            return {"insight": out["answer"], "sources": out.get("sources", []), "error": None}
        # Structured lookup by listing/model/year could be added here
        return {"insight": "Provide a natural-language query for resale guidance, or specify listing_id/model/year for structured lookup.", "sources": [], "error": None}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/export/pdf")
def export_pdf():
    """Placeholder: export results to PDF for clients or pitch decks. Frontend can generate client-side or call this with payload."""
    return {"message": "Use frontend Download PDF for chat/comparison/valuation reports. Server-side PDF generation can be added here."}
