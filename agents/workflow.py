from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any
from langchain_core.documents import Document
import logging

from .research_agent import ResearchAgent
from .verification_agent import VerificationAgent
from .relevance_checker import RelevanceChecker

logger = logging.getLogger(__name__)


# ---------------------------
# Agent State Definition
# ---------------------------
class AgentState(TypedDict):
    question: str
    documents: List[Document]
    draft_answer: str
    verification_report: str
    is_relevant: bool
    retriever: Any
    retrieval_scores: Dict  # NEW: Store retrieval scores


# ---------------------------
# Workflow Class
# ---------------------------
class AgentWorkflow:
    def __init__(self):
        self.researcher = ResearchAgent()
        self.verifier = VerificationAgent()
        self.relevance_checker = RelevanceChecker()
        self.compiled_workflow = self.build_workflow()

    # ---------------------------
    # Build LangGraph Workflow
    # ---------------------------
    def build_workflow(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("check_relevance", self._check_relevance_step)
        workflow.add_node("research", self._research_step)
        workflow.add_node("verify", self._verification_step)

        workflow.set_entry_point("check_relevance")

        workflow.add_conditional_edges(
            "check_relevance",
            self._decide_after_relevance_check,
            {
                "relevant": "research",
                "irrelevant": END
            }
        )

        workflow.add_edge("research", "verify")

        workflow.add_conditional_edges(
            "verify",
            self._decide_next_step,
            {
                "re_research": "research",
                "end": END
            }
        )

        return workflow.compile()

    # ---------------------------
    # Relevance Check
    # ---------------------------
    def _check_relevance_step(self, state: AgentState) -> Dict:
        retriever = state["retriever"]

        classification = self.relevance_checker.check(
            question=state["question"],
            retriever=retriever,
            k=20
        )

        if classification in ("CAN_ANSWER", "PARTIAL"):
            return {"is_relevant": True}

        return {
            "is_relevant": False,
            "draft_answer": (
                "This question is not related to the uploaded document(s), "
                "or there is insufficient information to answer it."
            )
        }

    def _decide_after_relevance_check(self, state: AgentState) -> str:
        decision = "relevant" if state["is_relevant"] else "irrelevant"
        logger.debug(f"Relevance decision: {decision}")
        return decision

    # ---------------------------
    # Full Pipeline Entry
    # ---------------------------
    def full_pipeline(self, question: str, retriever: Any):
        try:
            logger.info(f"Running pipeline for question: {question}")

            # Retrieve documents
            documents = retriever.invoke(question)
            
            # Get retrieval scores if available
            retrieval_scores = {}
            if hasattr(retriever, 'get_detailed_scores'):
                retrieval_scores = retriever.get_detailed_scores(question)

            logger.info(f"Retrieved {len(documents)} documents")

            initial_state: AgentState = {
                "question": question,
                "documents": documents,
                "draft_answer": "",
                "verification_report": "",
                "is_relevant": False,
                "retriever": retriever,
                "retrieval_scores": retrieval_scores  # NEW: Include scores
            }

            final_state = self.compiled_workflow.invoke(initial_state)
            
            # Combine verification report with retrieval scores
            verification_report = final_state.get("verification_report", "")
            if retrieval_scores and verification_report:
                # Enhance verification report with scores
                enhanced_report = self._enhance_report_with_scores(
                    verification_report, 
                    retrieval_scores
                )
            else:
                enhanced_report = verification_report

            return {
                "draft_answer": final_state.get("draft_answer", ""),
                "verification_report": enhanced_report,
                "retrieval_scores": retrieval_scores
            }

        except Exception as e:
            logger.exception("Workflow execution failed")
            raise e

    # ---------------------------
    # NEW: Enhance Report with Scores
    # ---------------------------
    def _enhance_report_with_scores(self, verification_report: str, retrieval_scores: Dict) -> str:
        """Add retrieval scores information to verification report"""
        enhanced_report = verification_report
        
        # Add retrieval scores section
        if retrieval_scores and 'combined_results' in retrieval_scores:
            enhanced_report += "\n\n--- RETRIEVAL SCORES ---\n"
            
            # Add BM25 scores if available
            if retrieval_scores.get('bm25_results'):
                enhanced_report += "\nBM25 Results:\n"
                for i, result in enumerate(retrieval_scores['bm25_results'][:3], 1):
                    enhanced_report += f"{i}. Score: {result.get('position_score', 0):.3f} | "
                    enhanced_report += f"Source: {result.get('source', 'unknown')[:50]}...\n"
            
            # Add Vector scores if available
            if retrieval_scores.get('vector_results'):
                enhanced_report += "\nVector Results:\n"
                for i, result in enumerate(retrieval_scores['vector_results'][:3], 1):
                    enhanced_report += f"{i}. Score: {result.get('position_score', 0):.3f} | "
                    enhanced_report += f"Source: {result.get('source', 'unknown')[:50]}...\n"
            
            # Add Combined scores
            if retrieval_scores.get('combined_results'):
                enhanced_report += "\nCombined Top Results:\n"
                for i, result in enumerate(retrieval_scores['combined_results'][:3], 1):
                    retriever_type = result.get('retriever_type', 'unknown').upper()
                    enhanced_report += f"{i}. [{retriever_type}] Score: {result.get('position_score', 0):.3f} | "
                    enhanced_report += f"Source: {result.get('source', 'unknown')[:50]}...\n"
        
        return enhanced_report

    # ---------------------------
    # Research Step
    # ---------------------------
    def _research_step(self, state: AgentState) -> Dict:
        logger.debug("Entering research step")
        result = self.researcher.generate(
            question=state["question"],
            documents=state["documents"]
        )
        return {"draft_answer": result["draft_answer"]}

    # ---------------------------
    # Verification Step
    # ---------------------------
    def _verification_step(self, state: AgentState) -> Dict:
        logger.debug("Entering verification step")
        result = self.verifier.check(
            answer=state["draft_answer"],
            documents=state["documents"]
        )
        return {"verification_report": result["verification_report"]}

    # ---------------------------
    # Decide Loop or End
    # ---------------------------
    def _decide_next_step(self, state: AgentState) -> str:
        report = state["verification_report"]

        if "Supported: NO" in report or "Relevant: NO" in report:
            logger.info("Verification failed → re-research")
            return "re_research"

        logger.info("Verification successful → end workflow")
        return "end"
