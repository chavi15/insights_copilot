import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DB_PATH, EVAL_DIR, GOLD_PATH, LOG_PATH
from src.evaluation import load_gold, run_evaluation, summarize, write_report
from src.llm import build_llm
from src.nl2sql import VARIANTS
from src.retrieval import get_retriever


def main():
    parser = argparse.ArgumentParser(description="Evaluate the three prompting variants on the gold set.")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    parser.add_argument("--limit", type=int, default=0, help="only run the first N gold questions")
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--provider", choices=["anthropic", "gemini"], default=None)
    parser.add_argument("--retriever", choices=["chroma", "keyword"], default=None)
    parser.add_argument("--no-log", action="store_true")
    args = parser.parse_args()

    gold = load_gold(GOLD_PATH)
    if args.limit:
        gold = gold[: args.limit]
    llm = build_llm(args.provider)
    retriever = get_retriever(args.retriever) if "retrieval_few_shot" in args.variants else None

    def progress(variant, index, total, correct):
        print(f"[{variant}] {index}/{total} {'ok' if correct else 'wrong'}", flush=True)

    records = run_evaluation(
        gold, llm, retriever, args.variants, args.db, log_path=None if args.no_log else LOG_PATH, progress=progress
    )
    meta = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model": f"{llm.name}:{llm.model}",
        "questions": len(gold),
        **{d: sum(1 for g in gold if g["difficulty"] == d) for d in ("easy", "medium", "hard")},
    }
    summary = write_report(records, EVAL_DIR / "REPORT.md", EVAL_DIR / "results.csv", meta)
    print(summary.to_string(index=False))
    print(f"cache hits: {llm.hits}, new calls: {llm.misses}")
    print(f"wrote {EVAL_DIR / 'REPORT.md'} and {EVAL_DIR / 'results.csv'}")


if __name__ == "__main__":
    main()
