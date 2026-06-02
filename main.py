import os
from dotenv import load_dotenv

load_dotenv()


def test_google_api() -> bool:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_MODEL", "gemini-1.5-flash"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
        response = llm.invoke("Reply with exactly two words: API connected")
        print(f"  Google API: {response.content.strip()}")
        return True
    except Exception as e:
        print(f"  Google API FAILED: {e}")
        return False


def test_langsmith() -> bool:
    try:
        from langsmith import Client
        client = Client(api_key=os.getenv("LANGCHAIN_API_KEY"))
        projects = list(client.list_projects())
        print(f"  LangSmith: connected — {len(projects)} project(s) found")
        return True
    except Exception as e:
        print(f"  LangSmith FAILED: {e}")
        return False


def main():
    print("=== Agentic AI RAG — Connection Tests ===\n")
    google_ok = test_google_api()
    langsmith_ok = test_langsmith()
    print(f"\nResult: Google={'OK' if google_ok else 'FAIL'}  LangSmith={'OK' if langsmith_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
