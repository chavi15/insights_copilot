import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import EVAL_DIR, LOG_PATH
from src.datasets import DATASETS
from src.evaluation import EvaluationStopped, load_gold, run_evaluation, summarize, write_report
from src.llm import build_llm, is_quota_error
from src.nl2sql import VARIANTS
from src.retrieval import get_retriever


def main():
    parser = argparse.ArgumentParser(description="Evaluate the three prompting variants on the gold set.")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    parser.add_argument("--limit", type=int, default=0, help="only run the first N gold questions")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="pharma")
    parser.add_argument("--db", default=None, help="database file; defaults to the dataset's own database")
    parser.add_argument("--provider", choices=["anthropic", "gemini"], default=None)
    parser.add_argument("--retriever", choices=["chroma", "keyword"], default=None)
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--stop-on-quota", action="store_true", help="stop at the first quota error instead of retrying")
    parser.add_argument("--pace", type=float, default=0.0, help="seconds to wait between questions")
    args = parser.parse_args()

    dataset = DATASETS[args.dataset]
    db_path = args.db or str(dataset.db_path)
    if not Path(db_path).exists():
        hint = {"retail": "python -m src.retail_load", "pharmacy": "python -m src.pharmacy_load"}.get(
            dataset.key, "python -m src.data_gen"
        )
        parser.error(f"{db_path} does not exist. Build it first with: {hint}")
    gold = load_gold(dataset.gold_path)
    if args.limit:
        gold = gold[: args.limit]
    llm = build_llm(args.provider, retry_quota=not args.stop_on_quota)
    retriever = get_retriever(args.retriever, dataset=dataset) if "retrieval_few_shot" in args.variants else None

    def progress(variant, index, total, correct):
        print(f"[{variant}] {index}/{total} {'ok' if correct else 'wrong'}", flush=True)

    def quota_stop(answer):
        if args.stop_on_quota and answer.error and is_quota_error(answer.error):
            return f"quota error, stopping: {answer.error[:200]}"
        return None

    try:
        records = run_evaluation(
            gold, llm, retriever, args.variants, db_path, log_path=None if args.no_log else LOG_PATH, progress=progress,
            dataset=dataset, stop_if=quota_stop, pause=(lambda: time.sleep(args.pace)) if args.pace else None,
        )
    except EvaluationStopped as stopped:
        print(f"STOPPED: {stopped}")
        print(f"{len(stopped.records)} of {len(gold) * len(args.variants)} questions finished; no report was written.")
        print(f"cache hits: {llm.hits}, new calls: {llm.misses}")
        sys.exit(2)
    meta = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model": f"{llm.name}:{llm.model}",
        "questions": len(gold),
        "title_suffix": f" ({dataset.label})" if dataset.key != "pharma" else "",
        **{d: sum(1 for g in gold if g["difficulty"] == d) for d in ("easy", "medium", "hard")},
    }
    report_path = EVAL_DIR / f"REPORT{dataset.report_suffix}.md"
    csv_path = EVAL_DIR / f"results{dataset.report_suffix}.csv"
    summary = write_report(records, report_path, csv_path, meta)
    print(summary.to_string(index=False))
    print(f"cache hits: {llm.hits}, new calls: {llm.misses}")
    print(f"wrote {report_path} and {csv_path}")


if __name__ == "__main__":
    main()
