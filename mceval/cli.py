"""``mceval`` command-line interface.

Commands:
    mceval run      — evaluate one adapter on LongMemEval
    mceval compare  — evaluate multiple adapters, print head-to-head table
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Callable

from mceval.adapters.base import MemoryAdapter
from mceval.adapters.bm25_baseline import BM25BaselineAdapter
from mceval.adapters.dense_baseline import DenseBaselineAdapter
from mceval.adapters.hybrid_rrf_baseline import HybridRRFBaselineAdapter
from mceval.adapters.memory_core import MemoryCoreAdapter
from mceval.datasets.longmemeval import load_longmemeval_oracle
from mceval.eval.runner import run_eval
from mceval.eval.scorer import QuestionResult
from mceval.eval.trace import TraceWriter

# Adapter registry: name -> factory accepting CLI kwargs
ADAPTERS: dict[str, Callable[..., MemoryAdapter]] = {
    "bm25": lambda **_: BM25BaselineAdapter(),
    "dense": lambda **_: DenseBaselineAdapter(),
    "hybrid-rrf": lambda **_: HybridRRFBaselineAdapter(),
    "memory-core": lambda base_url=None, api_key=None, **_: MemoryCoreAdapter(
        base_url=base_url, api_key=api_key
    ),
}


def _build_adapter(name: str, args: argparse.Namespace) -> MemoryAdapter:
    if name not in ADAPTERS:
        raise SystemExit(
            f"unknown adapter: {name!r}. "
            f"available: {sorted(ADAPTERS)}"
        )
    return ADAPTERS[name](
        base_url=getattr(args, "base_url", None),
        api_key=getattr(args, "api_key", None),
    )


def _progress_line(done: int, total: int, r: QuestionResult) -> None:
    hits = "".join("✓" if r.recall.get(k) else "✗" for k in sorted(r.recall))
    print(
        f"  [{done:3d}/{total}] {r.question_id[:32]:<32} "
        f"{r.question_type[:22]:<22} idx={r.n_indexed:3d}  {hits}  "
        f"({r.elapsed_s:.1f}s)",
        flush=True,
    )


def _print_summary(adapter_name: str, out) -> None:
    m = out.metrics
    ov = m.get("overall", {})
    print("")
    print("─" * 62)
    print(f"  {adapter_name} — n={m['n_total']} ({out.meta['elapsed_s']:.0f}s)")
    print(f"  Answerable: {m['n_answerable']}  |  Abstention: {m['n_abstention']}")
    print("─" * 62)
    parts = "  ".join(f"R@{k}={ov.get(f'recall@{k}',0):5.1f}%" for k in [1, 5, 10])
    print(f"  Overall (answerable)  {parts}")
    print("─" * 62)
    for qtype, x in m.get("by_type", {}).items():
        parts = "  ".join(f"@{k}={x.get(f'recall@{k}',0):5.1f}%" for k in [1, 5, 10])
        print(f"  {qtype:<30} n={x['n']:3d}  {parts}")
    abs_m = m.get("abstention", {})
    if abs_m:
        parts = "  ".join(f"@{k}={abs_m.get(f'precision@{k}',0):5.1f}%" for k in [1, 5, 10])
        print(f"  {'abstention (no-answer)':<30} n={m['n_abstention']:3d}  {parts}  [✗=correct]")
    print("─" * 62)


def _save_output(out, out_dir: Path, adapter_name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = out_dir / f"{adapter_name}_{ts}_n{out.meta['n_questions']}.json"

    payload = {
        "meta": out.meta,
        "metrics": out.metrics,
        "results": [_result_to_dict(r) for r in out.results],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return path


def _result_to_dict(r: QuestionResult) -> dict:
    d = asdict(r)
    d["recall"] = {str(k): v for k, v in r.recall.items()}
    return d


def cmd_run(args: argparse.Namespace) -> int:
    dataset = load_longmemeval_oracle(
        sample=args.sample,
        seed=args.seed,
        stratified=args.stratified,
    )
    tag = "stratified" if args.stratified else (f"seed={args.seed}" if args.seed is not None else "head")
    print(f"Loaded {len(dataset)} LongMemEval oracle questions ({tag})")

    adapter = _build_adapter(args.adapter, args)
    print(f"Running adapter: {adapter.name}  (workers={args.workers})")

    trace_writer = TraceWriter(args.trace) if args.trace else None
    try:
        out = run_eval(
            adapter=adapter,
            dataset=dataset,
            workers=args.workers,
            on_progress=_progress_line if args.verbose else None,
            trace_writer=trace_writer,
        )
    finally:
        if trace_writer:
            trace_writer.close()

    _print_summary(args.adapter, out)

    if args.out_dir:
        path = _save_output(out, Path(args.out_dir), args.adapter)
        print(f"\n  Saved → {path}")

    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    adapters = [a.strip() for a in args.adapters.split(",") if a.strip()]
    dataset = load_longmemeval_oracle(
        sample=args.sample,
        seed=args.seed,
        stratified=args.stratified,
    )
    tag = "stratified" if args.stratified else (f"seed={args.seed}" if args.seed is not None else "head")
    print(f"Loaded {len(dataset)} questions ({tag}). Comparing: {', '.join(adapters)}\n")

    rows = []
    for name in adapters:
        adapter = _build_adapter(name, args)
        out = run_eval(
            adapter=adapter,
            dataset=dataset,
            workers=args.workers,
            on_progress=None,
        )
        rows.append((name, out.metrics, out.meta["elapsed_s"]))
        if args.out_dir:
            _save_output(out, Path(args.out_dir), name)

    print("\n" + "─" * 72)
    print(f"  {'adapter':<18} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'n':>6} {'time':>8}")
    print("─" * 72)
    for name, m, t in rows:
        ov = m.get("overall", {})
        print(
            f"  {name:<18} "
            f"{ov.get('recall@1',0):>7.1f}% "
            f"{ov.get('recall@5',0):>7.1f}% "
            f"{ov.get('recall@10',0):>7.1f}% "
            f"{m['n_answerable']:>6} "
            f"{t:>7.0f}s"
        )
    print("─" * 72)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mceval", description="memory-core-eval CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    # shared
    def _add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--sample", type=int, default=None, help="Eval only N questions (see --seed / --stratified)")
        sp.add_argument("--seed", type=int, default=None,
                        help="Shuffle with this seed before sampling. Required for representative small samples.")
        sp.add_argument("--stratified", action="store_true",
                        help="Sample proportionally across question_type (uses --seed for tie-break, default 0).")
        sp.add_argument("--workers", type=int, default=4, help="Parallel question workers")
        sp.add_argument("--base-url", default=None, help="Override adapter base URL (e.g. self-hosted Memory Core)")
        sp.add_argument("--api-key", default=None, help="API key for hosted adapters")
        sp.add_argument("--out-dir", default="results", help="Directory for JSON output")

    run = sub.add_parser("run", help="Run one adapter")
    run.add_argument("--adapter", required=True, choices=sorted(ADAPTERS))
    run.add_argument("--trace", default=None, help="Write per-question JSONL trace to this path")
    run.add_argument("--verbose", action="store_true")
    _add_common(run)
    run.set_defaults(func=cmd_run)

    cmp_ = sub.add_parser("compare", help="Head-to-head multi-adapter compare")
    cmp_.add_argument("--adapters", required=True, help="Comma-separated adapter names")
    _add_common(cmp_)
    cmp_.set_defaults(func=cmd_compare)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
