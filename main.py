"""
Medical Rehab Agent System — Main Entry Point

Usage:
  # CLI mode (default)
  python main.py

  # API server mode
  python main.py --serve

  # Custom patient input
  python main.py --age 8 --symptoms "发音不清，注意力不集中" --history "无"
"""

import argparse
import json
import sys

from rag.retriever import RAGRetriever
from agents.dispatcher import DispatcherAgent
from utils.report import to_structured_json, to_pdf


def run_cli(patient_data: dict, generate_pdf: bool = False):
    """Run the full pipeline from CLI."""
    print("=" * 60)
    print("  Medical Rehab Agent System")
    print("=" * 60)

    # ── Check LLM availability ──────────────────────────────────
    from config import config
    from agents.base import BaseAgent

    print(f"\n[Config] Provider={config.LLM_PROVIDER}, Model={config.LLM_MODEL}")
    if config.LLM_BASE_URL:
        print(f"[Config] Base URL={config.LLM_BASE_URL}")

    if not config.LLM_API_KEY or config.LLM_API_KEY == "not-needed" and config.LLM_PROVIDER != "local":
        print("\n[Warning] No LLM_API_KEY configured. Run 'python3 check_llm.py' to verify.")
        print("[Info]    Without a real LLM, agents will fail. Use 'python3 demo.py' for mock mode.\n")

    llm_check = BaseAgent.test_connection()
    if not llm_check["ok"]:
        print(f"[Error] LLM connection failed: {llm_check['error']}")
        print("[Info]  Please check your .env file. See LLM_GUIDE.py for setup instructions.")
        print("[Info]  Or run 'python3 demo.py' to try with mock data.\n")
        sys.exit(1)
    print(f"[LLM] Connection OK — {llm_check.get('model', config.LLM_MODEL)}")

    # ── Initialize RAG ──────────────────────────────────────────
    print("\n[RAG] Initializing knowledge base...")
    retriever = RAGRetriever()
    retriever.initialize()

    # ── Initialize Dispatcher ───────────────────────────────────
    print("[Agent] Initializing agent pipeline...")
    dispatcher = DispatcherAgent(retriever)

    # ── Run Pipeline ────────────────────────────────────────────
    print(f"\n[Input] Patient: age={patient_data['age']}, symptoms={patient_data['symptoms']}")
    print("[Pipeline] Running SymptomAnalysis → RiskAssessment → RehabPlan...\n")

    result = dispatcher.run(patient_data)

    # ── Output ──────────────────────────────────────────────────
    structured_output = to_structured_json(result)
    print("=" * 60)
    print("  Structured Output")
    print("=" * 60)
    print(structured_output)

    # ── Optional PDF ────────────────────────────────────────────
    if generate_pdf:
        pdf_path = to_pdf(result, "report.pdf")
        print(f"\n[PDF] Report saved to: {pdf_path}")

    # ── Full Raw Output ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Full Agent Output (raw)")
    print("=" * 60)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    return result


def run_server():
    """Start the FastAPI server."""
    import uvicorn
    from config import config

    print(f"Starting API server on {config.API_HOST}:{config.API_PORT}...")
    uvicorn.run(
        "api.server:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Medical Rehab Agent System")
    parser.add_argument("--serve", action="store_true", help="Start FastAPI server")
    parser.add_argument("--age", type=int, default=10, help="Patient age")
    parser.add_argument("--symptoms", type=str, default="发音不清，语言表达困难，注意力不集中", help="Symptom description")
    parser.add_argument("--history", type=str, default="无重大疾病史", help="Medical history")
    parser.add_argument("--pdf", action="store_true", help="Generate PDF report")

    args = parser.parse_args()

    if args.serve:
        run_server()
    else:
        patient_data = {
            "age": args.age,
            "symptoms": args.symptoms,
            "history": args.history,
        }
        run_cli(patient_data, generate_pdf=args.pdf)


if __name__ == "__main__":
    main()
