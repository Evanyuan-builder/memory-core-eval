"""End-to-end QA-accuracy eval — SAME retrieval path as runner.py, but instead
of scoring Recall@k it feeds the retrieved memories to an LLM reader, generates
an answer, and an LLM judge grades it against the gold answer.

This measures ANSWER ACCURACY (retrieval + reading) — the metric TiMEM / Mem0 /
LongMemEval papers report — NOT Recall@k. The judge here is our own LLM grader
(DeepSeek by default), not the official LongMemEval GPT-4 grader, so the numbers
are for *same-axis relative comparison*; label them as such when reporting.

Usage:
    LLM_API_KEY=... .venv/bin/python -m mceval.eval.qa_runner \
        --adapter memory-core --dataset locomo --sample 100 --stratified --seed 0 \
        --workers 2 --top-k 10 --out-dir baselines_qa
"""
from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

from mceval.adapters.base import Memory, MemoryAdapter, Turn
from mceval.cli import ADAPTERS
from mceval.datasets.locomo import load_locomo
from mceval.datasets.longmemeval import (
    is_abstention,
    iter_turns,
    load_longmemeval,
    parse_longmemeval_date,
)

# ── LLM client (OpenAI-compatible; DeepSeek default) ────────────────────────

class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 60,
                 thinking: bool = False):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.thinking = thinking  # True = let the model reason (DeepSeek V4 default-on)

    def chat(self, system: str, user: str, max_tokens: int = 512) -> str:
        url = f"{self.base_url}/chat/completions"
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        if not self.thinking:
            # DeepSeek V4 fast path; unknown-field-safe on other providers.
            body["thinking"] = {"type": "disabled"}
        elif self.thinking and "deepseek" in self.model.lower():
            body["max_tokens"] = max(max_tokens, 2048)  # leave room for reasoning tokens
        last = None
        for attempt in range(4):
            try:
                r = requests.post(
                    url, json=body,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=self.timeout,
                )
                if r.status_code == 200:
                    return r.json()["choices"][0]["message"]["content"].strip()
                last = f"{r.status_code}: {r.text[:200]}"
            except Exception as e:  # noqa: BLE001
                last = str(e)
            time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"LLM call failed after retries: {last}")


# v1 = original conservative reader (bare snippets). Kept so the pre-2026-06-18
# apples-to-apples matrix (memory-core 26/49, TiMEM-LoCoMo 29, bm25, hybrid) stays
# exactly reproducible. v2 below is the improved reader.
READER_SYS_V1 = (
    "You answer a question using ONLY the provided memory snippets from past "
    "conversations. Answer concisely and directly with just the fact asked for. "
    "If the snippets do not contain enough information, reply exactly: I don't know."
)

READER_SYS = (
    "You answer a question using the provided memory snippets from past "
    "conversations. Each snippet may be prefixed with its speaker and date in "
    "brackets, e.g. [Caroline · 2023-05-01] — use these to attribute who said "
    "what and to resolve temporal references (\"last year\", \"when we met\"). "
    "Reason across the snippets to infer the answer even when it is not stated "
    "verbatim; combine multiple snippets when the answer is spread across them. "
    "Answer concisely with just the fact asked for — a name, date, number, or "
    "short phrase. Only when the snippets contain nothing relevant at all, "
    "reply exactly: I don't know."
)

JUDGE_SYS = (
    "You grade whether a model's answer matches the gold answer to a question. "
    "The answer is CORRECT if it conveys the same key information as the gold "
    "answer, even if phrased differently or with extra detail. Numerical, "
    "temporal, and named-entity facts must match. Otherwise WRONG. "
    "Respond with exactly one word: CORRECT or WRONG."
)

_REFUSAL = (
    "i don't know", "i dont know", "no information", "not mentioned",
    "cannot determine", "can't determine", "don't have", "do not have",
    "no relevant", "unable to",
)


def _fmt_date(d) -> str:
    """Best-effort YYYY-MM-DD from a datetime / ISO string / None."""
    if d is None:
        return ""
    if isinstance(d, datetime):
        return d.strftime("%Y-%m-%d")
    s = str(d).strip()
    return s[:10] if s else ""


def _format_snippet(content: str, meta_by_content: dict[str, tuple[str, str]]) -> str:
    """Prefix a retrieved snippet with [speaker · date] when we know them.

    The map is built at ingest from the question's own turns, so it only hits
    for turn-level retrievers that return verbatim turn content (memory-core,
    bm25, hybrid). Consolidated/rewritten memories (e.g. TiMEM) won't match and
    stay bare — which faithfully reflects what each system hands a downstream
    reader."""
    meta = meta_by_content.get(content)
    if not meta:
        return content
    speaker, date = meta
    tag = " · ".join(p for p in (speaker, date) if p)
    return f"[{tag}] {content}" if tag else content


def _reader_prompt(question: str, snippets: list[str]) -> str:
    ctx = "\n".join(f"[{i + 1}] {s}" for i, s in enumerate(snippets)) or "(none)"
    return f"Memory snippets:\n{ctx}\n\nQuestion: {question}\n\nAnswer:"


def _judge_prompt(question: str, gold: str, predicted: str) -> str:
    return (
        f"Question: {question}\nGold answer: {gold}\n"
        f"Model answer: {predicted}\n\nGrade (CORRECT or WRONG):"
    )


@dataclass
class QAResult:
    question_id: str
    question_type: str
    abstention: bool
    correct: bool
    n_indexed: int
    n_retrieved: int
    question: str = ""
    gold: str = ""
    predicted: str = ""
    judge_raw: str = ""
    error: Optional[str] = None
    elapsed_s: float = 0.0


def _evaluate_one(adapter: MemoryAdapter, item: dict, top_k: int,
                  reader: LLMClient, judge: LLMClient, enrich: bool = True,
                  reader_sys: str = READER_SYS) -> QAResult:
    qid = item["question_id"]
    question = item["question"]
    q_type = item.get("question_type", "unknown")
    abstention = is_abstention(item)
    gold = item.get("answer", "") or ""
    ns = f"mcqa-{uuid.uuid4().hex[:12]}"
    t0 = time.perf_counter()

    adapter.reset(ns)
    qdate = parse_longmemeval_date(item.get("question_date"))

    stored = 0
    meta_by_content: dict[str, tuple[str, str]] = {}
    for sid, s_idx, t_idx, role, content, sdate in iter_turns(item):
        try:
            adapter.store(ns, Turn(content=content, role=role, session_id=sid,
                                   turn_idx=t_idx, session_idx=s_idx, timestamp=sdate))
            stored += 1
            if enrich:
                meta_by_content[content] = (role or "", _fmt_date(sdate))
        except Exception:  # noqa: BLE001
            continue

    def _fin(correct, predicted="", judge_raw="", n_ret=0, err=None):
        try:
            adapter.reset(ns)
        except Exception:  # noqa: BLE001
            pass
        return QAResult(qid, q_type, abstention, correct, stored, n_ret,
                        question, gold, predicted, judge_raw, err,
                        round(time.perf_counter() - t0, 2))

    if stored == 0:
        return _fin(False, err="nothing_indexed")

    retrieved: list[Memory] = adapter.search(ns, question, top_k=top_k, as_of_date=qdate)
    snippets = [_format_snippet(m.content, meta_by_content) for m in retrieved][:top_k]

    try:
        predicted = reader.chat(reader_sys, _reader_prompt(question, snippets))
    except Exception as e:  # noqa: BLE001
        return _fin(False, n_ret=len(retrieved), err=f"reader:{e}")

    pl = predicted.lower()
    refused = any(p in pl for p in _REFUSAL)

    if abstention:
        # gold = should abstain; correct iff model refused to answer
        return _fin(refused, predicted, "ABSTAIN_HEURISTIC", len(retrieved))
    if refused:
        return _fin(False, predicted, "REFUSED", len(retrieved))

    try:
        jr = judge.chat(JUDGE_SYS, _judge_prompt(question, gold, predicted), max_tokens=8)
    except Exception as e:  # noqa: BLE001
        return _fin(False, predicted, n_ret=len(retrieved), err=f"judge:{e}")

    correct = jr.strip().upper().startswith("CORRECT")
    return _fin(correct, predicted, jr, len(retrieved))


def run_qa_eval(adapter, dataset, top_k, reader, judge, workers, verbose=False,
                enrich=True, reader_sys=READER_SYS):
    t_start = time.perf_counter()
    n = len(dataset)
    completed: dict[int, QAResult] = {}

    def _task(idx_item):
        idx, item = idx_item
        return idx, _evaluate_one(adapter, item, top_k, reader, judge,
                                  enrich=enrich, reader_sys=reader_sys)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_task, (i, it)): i for i, it in enumerate(dataset)}
        done = 0
        for fut in as_completed(futures):
            idx, res = fut.result()
            completed[idx] = res
            done += 1
            if verbose:
                mark = "✓" if res.correct else ("·" if res.error else "✗")
                print(f"  [{done:3d}/{n}] {res.question_id[:30]:<30} "
                      f"{res.question_type[:18]:<18} {mark} "
                      f"pred={res.predicted[:40]!r}" + (f" ERR={res.error}" if res.error else ""),
                      flush=True)

    results = [completed[i] for i in range(n)]
    elapsed = time.perf_counter() - t_start

    answerable = [r for r in results if not r.abstention]
    abst = [r for r in results if r.abstention]
    errs = [r for r in results if r.error]

    def acc(rs):
        return round(100.0 * sum(r.correct for r in rs) / len(rs), 1) if rs else 0.0

    by_type: dict[str, dict] = {}
    types = sorted({r.question_type for r in results})
    for t in types:
        rs = [r for r in results if r.question_type == t]
        by_type[t] = {"n": len(rs), "accuracy": acc(rs)}

    metrics = {
        "n_total": n,
        "n_answerable": len(answerable),
        "n_abstention": len(abst),
        "n_error": len(errs),
        "overall_accuracy": acc(results),
        "answerable_accuracy": acc(answerable),
        "abstention_accuracy": acc(abst),
        "by_type": by_type,
    }
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "adapter": getattr(adapter, "name", type(adapter).__name__),
        "metric": "qa_accuracy (LLM reader + LLM judge; NOT official LongMemEval grader)",
        "top_k": top_k,
        "reader_model": reader.model,
        "judge_model": judge.model,
        "workers": workers,
        "elapsed_s": round(elapsed, 1),
    }
    return meta, metrics, results


def main() -> int:
    ap = argparse.ArgumentParser(description="End-to-end QA-accuracy eval (same axis as TiMEM/Mem0 papers)")
    ap.add_argument("--adapter", required=True, choices=sorted(ADAPTERS))
    ap.add_argument("--dataset", default="locomo", choices=["longmemeval", "locomo"])
    ap.add_argument("--split", default="s", choices=["m", "oracle", "s"])
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stratified", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--base-url", default=None, help="adapter base url (memory-core api)")
    ap.add_argument("--api-key", default=None, help="adapter api key")
    ap.add_argument("--llm-base-url", default=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1"))
    ap.add_argument("--llm-api-key", default=os.environ.get("LLM_API_KEY"))
    ap.add_argument("--reader-model", default=os.environ.get("LLM_MODEL", "deepseek-v4-flash"))
    ap.add_argument("--judge-model", default=None, help="defaults to reader model")
    ap.add_argument("--reader-thinking", action="store_true",
                    help="let the reader model reason (DeepSeek V4 thinking-on); judge stays fast")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--bare-snippets", action="store_true",
                    help="disable [speaker · date] enrichment")
    ap.add_argument("--reader-prompt", choices=["v1", "v2"], default="v2",
                    help="v1 = original conservative reader (forces bare); "
                         "v2 = improved reader (default). v1 reproduces the pre-2026-06-18 matrix.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    reader_sys = READER_SYS_V1 if args.reader_prompt == "v1" else READER_SYS
    # v1 is the original behavior — always bare. v2 enriches unless --bare-snippets.
    enrich = (args.reader_prompt == "v2") and (not args.bare_snippets)

    if not args.llm_api_key:
        raise SystemExit("No LLM key: pass --llm-api-key or set LLM_API_KEY")

    if args.dataset == "longmemeval":
        dataset = load_longmemeval(split=args.split, sample=args.sample,
                                   seed=args.seed, stratified=args.stratified)
        display = f"LongMemEval-{args.split}"
        dataset_tag = f"longmemeval-{args.split}"
    else:
        dataset = load_locomo(sample=args.sample, seed=args.seed, stratified=args.stratified)
        display = "LoCoMo"
        dataset_tag = "locomo"

    adapter = ADAPTERS[args.adapter](base_url=args.base_url, api_key=args.api_key)
    reader = LLMClient(args.llm_base_url, args.llm_api_key, args.reader_model,
                       timeout=180 if args.reader_thinking else 60,
                       thinking=args.reader_thinking)
    judge = LLMClient(args.llm_base_url, args.llm_api_key, args.judge_model or args.reader_model)

    print(f"Loaded {len(dataset)} {display} questions | adapter={adapter.name} "
          f"top_k={args.top_k} reader={reader.model} judge={judge.model} workers={args.workers}")

    meta, metrics, results = run_qa_eval(adapter, dataset, args.top_k, reader, judge,
                                         args.workers, verbose=args.verbose,
                                         enrich=enrich, reader_sys=reader_sys)
    meta["snippet_enrichment"] = "speaker+date" if enrich else "bare"
    meta["reader_prompt"] = args.reader_prompt
    meta["reader_thinking"] = bool(args.reader_thinking)

    print("\n" + "─" * 62)
    print(f"  {args.adapter} — {display} — n={metrics['n_total']} ({meta['elapsed_s']:.0f}s)")
    print(f"  metric: QA ACCURACY (our LLM judge, not official grader)")
    print("─" * 62)
    print(f"  Overall accuracy        {metrics['overall_accuracy']:5.1f}%   (n={metrics['n_total']})")
    print(f"  Answerable accuracy     {metrics['answerable_accuracy']:5.1f}%   (n={metrics['n_answerable']})")
    if metrics["n_abstention"]:
        print(f"  Abstention accuracy     {metrics['abstention_accuracy']:5.1f}%   (n={metrics['n_abstention']})")
    if metrics["n_error"]:
        print(f"  Errors                  {metrics['n_error']}")
    print("─" * 62)
    for t, x in metrics["by_type"].items():
        print(f"  {t:<28} n={x['n']:3d}  acc={x['accuracy']:5.1f}%")
    print("─" * 62)

    if args.out_dir:
        outd = Path(args.out_dir)
        outd.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = outd / f"qa_{args.adapter}_{dataset_tag}_{ts}_n{metrics['n_total']}.json"
        path.write_text(json.dumps(
            {"meta": meta, "metrics": metrics, "results": [asdict(r) for r in results]},
            indent=2, ensure_ascii=False))
        print(f"\n  Saved → {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
