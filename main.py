"""
Smoke-test runner — verifies all services are reachable and runs one end-to-end
agent query so you can confirm the full pipeline works before starting the API.

Usage:
    uv run python main.py
"""

import os
from dotenv import load_dotenv

load_dotenv()


# ── Service connectivity checks ───────────────────────────────────────────────

def check_groq() -> bool:
    try:
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            api_key=os.getenv("GROQ_API_KEY"),
            max_tokens=20,
        )
        resp = llm.invoke("Reply with the single word: connected")
        print(f"  Groq        : {str(resp.content).strip()[:60]}")
        return True
    except Exception as e:
        print(f"  Groq        : FAIL — {e}")
        return False


def check_qdrant() -> bool:
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=10,
        )
        collections = [c.name for c in client.get_collections().collections]
        print(f"  Qdrant      : OK — collections: {collections}")
        return True
    except Exception as e:
        print(f"  Qdrant      : FAIL — {e}")
        return False


def check_redis() -> bool:
    try:
        from app.memory.session import ping
        ok = ping()
        print(f"  Redis       : {'OK' if ok else 'FAIL'}")
        return ok
    except Exception as e:
        print(f"  Redis       : FAIL — {e}")
        return False


def check_postgres() -> bool:
    try:
        from app.memory.store import ping
        ok = ping()
        print(f"  PostgreSQL  : {'OK' if ok else 'FAIL'}")
        return ok
    except Exception as e:
        print(f"  PostgreSQL  : FAIL — {e}")
        return False


def check_langsmith() -> bool:
    key = os.getenv("LANGCHAIN_API_KEY")
    if not key:
        print("  LangSmith   : skipped (LANGCHAIN_API_KEY not set)")
        return True
    try:
        from langsmith import Client
        projects = list(Client(api_key=key).list_projects())
        print(f"  LangSmith   : OK — {len(projects)} project(s)")
        return True
    except Exception as e:
        print(f"  LangSmith   : FAIL — {e}")
        return False


# ── End-to-end agent smoke test ───────────────────────────────────────────────

def run_agent_test() -> bool:
    print("\nRunning end-to-end agent test...")
    try:
        from app.agents.graph.stateful import run
        result = run(
            question="What is the difference between cash flow from operations and net income?",
            session_id="smoke-test-session",
        )
        print(f"  Plan        : {result['plan']}")
        print(f"  Iterations  : {result['iteration_count']}")
        print(f"  Confidence  : {result['confidence']:.2f}")
        print(f"  Answer      :\n{result['answer'][:400]}...")
        print(f"  Sources     : {len(result['sources'])} chunk(s) cited")
        return True
    except Exception as e:
        print(f"  Agent test  : FAIL — {e}")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== Agentic AI RAG — System Check ===\n")

    groq_ok  = check_groq()
    qdrant_ok = check_qdrant()
    redis_ok  = check_redis()
    pg_ok     = check_postgres()
    ls_ok     = check_langsmith()

    all_ok = groq_ok and qdrant_ok
    print(f"\nServices: Groq={'OK' if groq_ok else 'FAIL'}  "
          f"Qdrant={'OK' if qdrant_ok else 'FAIL'}  "
          f"Redis={'OK' if redis_ok else 'FAIL'}  "
          f"PG={'OK' if pg_ok else 'FAIL'}")

    if all_ok:
        run_agent_test()
    else:
        print("\nSkipping agent test — fix service connections first.")


if __name__ == "__main__":
    main()
