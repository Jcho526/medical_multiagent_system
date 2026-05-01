"""Dispatcher Agent — orchestrates the multi-agent pipeline."""

from typing import Dict, Any
from rag.retriever import RAGRetriever
from agents.symptom_analyzer import SymptomAnalysisAgent
from agents.risk_assessor import RiskAssessmentAgent
from agents.rehab_planner import RehabPlanAgent


class DispatcherAgent:
    """Coordinates the analysis pipeline:

    Patient Input → RAG Retrieval → SymptomAnalysis → RiskAssessment → RehabPlan → Output
    """

    name = "DispatcherAgent"

    def __init__(self, rag_retriever: RAGRetriever):
        self.rag = rag_retriever
        self.symptom_agent = SymptomAnalysisAgent()
        self.risk_agent = RiskAssessmentAgent()
        self.rehab_agent = RehabPlanAgent()

    def run(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the full pipeline and return structured results."""

        # ── Step 1: RAG Retrieval ────────────────────────────────
        query = self._build_query(patient_data)
        rag_results = self.rag.retrieve(query)
        rag_context = self.rag.format_context(rag_results)

        # ── Step 2: Symptom Analysis ─────────────────────────────
        symptom_analysis = self.symptom_agent.run(patient_data, rag_context)

        # ── Step 3: Risk Assessment ──────────────────────────────
        risk_assessment = self.risk_agent.run(symptom_analysis, patient_data)

        # ── Step 4: Rehabilitation Plan ──────────────────────────
        rehab_plan = self.rehab_agent.run(
            patient_data, symptom_analysis, risk_assessment, rag_context
        )

        # ── Assemble Final Output ────────────────────────────────
        return {
            "patient": patient_data,
            "symptom_analysis": symptom_analysis,
            "risk_assessment": risk_assessment,
            "rehab_plan": rehab_plan,
            "rag_sources": [
                {"id": r.get("id"), "disease": r.get("disease")}
                for r in rag_results
            ],
        }

    @staticmethod
    def _build_query(patient_data: Dict) -> str:
        """Construct a retrieval query from patient data."""
        parts = []
        if "symptoms" in patient_data:
            parts.append(patient_data["symptoms"])
        if "age" in patient_data:
            parts.append(f"儿童 年龄{patient_data['age']}")
        return " ".join(parts)
