from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.orchestrator import DocumentQAAgent


EVAL_SUITES = {
    "gbt": Path(__file__).parent / "golden_qa_gbt.json",
    "agent": Path(__file__).parent / "golden_qa_agent_sample.json",
    "all": Path(__file__).parent / "golden_qa.json",
}


@dataclass
class EvalCase:
    id: str
    doc_id: str
    type: str
    question: str
    expect_answered: bool
    expected_keywords: list[str]
    expected_clause_ids: list[str]
    gold_pages: list[int]


@dataclass
class EvalResult:
    case: EvalCase
    passed: bool
    detail: str
    retrieval_hit: bool
    answer_ok: bool
    citation_ok: bool
    status: str


def extract_digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def keyword_match(answer: str | None, keywords: list[str]) -> bool:
    if not answer:
        return False
    answer_digits = extract_digits(answer)
    for keyword in keywords:
        if any(ch.isdigit() for ch in keyword):
            if extract_digits(keyword) not in answer_digits:
                return False
        elif keyword not in answer:
            return False
    return True


def load_cases(path: Path) -> list[EvalCase]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases: list[EvalCase] = []
    for item in data:
        cases.append(
            EvalCase(
                id=item["id"],
                doc_id=item.get("doc_id", ""),
                type=item["type"],
                question=item["question"],
                expect_answered=item["expect_answered"],
                expected_keywords=item.get("expected_keywords", []),
                expected_clause_ids=item.get("expected_clause_ids", []),
                gold_pages=item.get("gold_pages", []),
            )
        )
    return cases


def load_combined_cases() -> list[EvalCase]:
    cases: list[EvalCase] = []
    for name in ("gbt", "agent"):
        cases.extend(load_cases(EVAL_SUITES[name]))
    return cases


def evaluate_case(agent: DocumentQAAgent, case: EvalCase) -> EvalResult:
    doc_ids = [case.doc_id] if case.doc_id else None
    response = agent.ask(case.question, doc_ids=doc_ids)

    retrieval_hit = False
    if case.gold_pages:
        retrieved_pages = {int(hit.metadata.get("page", 0)) for hit in response.retrieved_chunks}
        retrieval_hit = any(page in retrieved_pages for page in case.gold_pages)
    elif case.expected_clause_ids:
        retrieved_clauses = {hit.metadata.get("clause_id") for hit in response.retrieved_chunks}
        retrieval_hit = any(clause in retrieved_clauses for clause in case.expected_clause_ids)
    else:
        retrieval_hit = True

    if not case.expect_answered:
        answer_ok = response.status == "refused"
        citation_ok = True
        passed = answer_ok
        detail = "refused correctly" if answer_ok else "should refuse but answered"
        return EvalResult(case, passed, detail, retrieval_hit, answer_ok, citation_ok, response.status)

    answer_ok = response.status == "answered" and keyword_match(response.answer, case.expected_keywords)
    citation_ok = bool(response.citations)
    if case.expected_clause_ids and response.citations:
        citation_ok = any(
            citation.clause_id in case.expected_clause_ids for citation in response.citations if citation.clause_id
        )
    if case.doc_id and response.citations:
        doc_ids_in_citations = {citation.doc_id for citation in response.citations if citation.doc_id}
        if doc_ids_in_citations:
            citation_ok = citation_ok and case.doc_id in doc_ids_in_citations
    passed = answer_ok and citation_ok
    detail = "ok" if passed else f"answer_ok={answer_ok}, citation_ok={citation_ok}, status={response.status}"
    return EvalResult(case, passed, detail, retrieval_hit, answer_ok, citation_ok, response.status)


def print_report(title: str, results: list[EvalResult]) -> None:
    total = len(results)
    passed = sum(1 for result in results if result.passed)
    retrieval_hits = sum(1 for result in results if result.retrieval_hit)
    answerable = [result for result in results if result.case.expect_answered]
    unanswerable = [result for result in results if not result.case.expect_answered]

    answer_acc = sum(1 for result in answerable if result.answer_ok) / max(len(answerable), 1)
    citation_acc = sum(1 for result in answerable if result.citation_ok) / max(len(answerable), 1)
    refusal_acc = sum(1 for result in unanswerable if result.answer_ok) / max(len(unanswerable), 1)
    recall_at5 = retrieval_hits / max(total, 1)

    print(f"=== {title} ===")
    print(f"Total Cases:         {total}")
    print(f"Overall Pass Rate:   {passed / total:.1%} ({passed}/{total})")
    print(f"Retrieval Recall@5:  {recall_at5:.1%}")
    print(f"Answer Accuracy:     {answer_acc:.1%}")
    print(f"Citation Accuracy:   {citation_acc:.1%}")
    if unanswerable:
        print(f"Refusal Accuracy:    {refusal_acc:.1%} ({sum(1 for r in unanswerable if r.answer_ok)}/{len(unanswerable)})")
    print("-" * 36)
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"{result.case.id} [{status}] {result.case.type} - {result.detail}")


def run_suite(agent: DocumentQAAgent, suite: str) -> tuple[list[EvalResult], int]:
    if suite == "combined":
        cases = load_combined_cases()
        title = "Combined Evaluation Report"
    else:
        cases = load_cases(EVAL_SUITES[suite])
        doc_label = cases[0].doc_id if cases else suite
        title = f"Evaluation Report: {doc_label}"

    results = [evaluate_case(agent, case) for case in cases]
    print_report(title, results)
    print()
    passed = sum(1 for result in results if result.passed)
    return results, passed


def run_eval(suite: str = "all") -> int:
    agent = DocumentQAAgent()

    if suite == "all":
        all_passed = 0
        all_total = 0
        for suite_name in ("gbt", "agent"):
            results, passed = run_suite(agent, suite_name)
            all_passed += passed
            all_total += len(results)
        print("=== Summary ===")
        print(f"Overall Pass Rate: {all_passed / all_total:.1%} ({all_passed}/{all_total})")
        return 0 if all_passed == all_total else 1

    suite_key = "combined" if suite == "combined" else suite
    _, passed = run_suite(agent, suite_key)
    cases = load_combined_cases() if suite == "combined" else load_cases(EVAL_SUITES[suite])
    return 0 if passed == len(cases) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run document QA evaluation")
    parser.add_argument(
        "--suite",
        choices=["gbt", "agent", "combined", "all"],
        default="all",
        help="Evaluation suite: gbt, agent, combined (both), or all (print all reports)",
    )
    args = parser.parse_args(argv)
    return run_eval(args.suite)


if __name__ == "__main__":
    raise SystemExit(main())
