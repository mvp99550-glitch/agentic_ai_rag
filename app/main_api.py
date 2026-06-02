"""FastAPI application entry point."""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from dotenv import load_dotenv

load_dotenv()

from app.api.routes import query as query_route
from app.memory.store import setup_tables, ping as pg_ping
from app.memory.session import ping as redis_ping

# Rate limiter — 20 requests/minute per IP
limiter = Limiter(key_func=get_remote_address, default_limits=["20/minute"])

app = FastAPI(
    title="Agentic AI RAG",
    description="Financial document RAG API",
    version="0.1.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS — allows Streamlit + future AWS API Gateway to call this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    setup_tables()
    print(f"  PostgreSQL : {'OK' if pg_ping()    else 'FAIL'}")
    print(f"  Redis      : {'OK' if redis_ping() else 'FAIL'}")


@app.get("/health")
def health():
    return {
        "status":     "ok",
        "postgres":   pg_ping(),
        "redis":      redis_ping(),
        "llm_ready":  False,
    }


app.include_router(query_route.router, prefix="/api/v1")
