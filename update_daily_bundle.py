import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

from backtest_optimize import load_config
from update_manual_hist import _clear_post_update_disk_caches, _post_refresh_prewarm, _write_refresh_signal


def _run(cmd: List[str]) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(Path(__file__).resolve().parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="一键更新日线 + 必盈资金流向 + 清缓存预热")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--universe-file", default="", help="Override universe file, default from config")
    parser.add_argument("--trade-calendar-file", default="data/trade_calendar.csv")
    parser.add_argument("--manual-workers", type=int, default=8)
    parser.add_argument("--manual-sleep", type=float, default=0.02)
    parser.add_argument("--fundflow-workers", type=int, default=4)
    parser.add_argument("--fundflow-sleep", type=float, default=0.1)
    parser.add_argument("--manual-full", action="store_true", help="日线全量更新，不只更新陈旧数据")
    parser.add_argument("--fundflow-full", action="store_true", help="资金流全量更新，不只更新陈旧数据")
    parser.add_argument("--manual-require-columns", default="")
    parser.add_argument("--fundflow-require-columns", default="")
    parser.add_argument("--prewarm-api-base", default="http://127.0.0.1:8000")
    parser.add_argument("--prewarm-timeout", type=int, default=20)
    parser.add_argument("--skip-prewarm", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    universe_file = str(args.universe_file or cfg.get("universe", {}).get("universe_file") or "./data/universe.csv")

    py = sys.executable

    manual_cmd = [
        py,
        "update_manual_hist.py",
        "--config",
        args.config,
        "--universe-file",
        universe_file,
        "--workers",
        str(args.manual_workers),
        "--sleep",
        str(args.manual_sleep),
        "--skip-post-refresh",
    ]
    if not args.manual_full:
        manual_cmd.append("--only-stale")
    if args.manual_require_columns.strip():
        manual_cmd.extend(["--require-columns", args.manual_require_columns.strip()])

    fundflow_cmd = [
        py,
        "update_biying_fundflow.py",
        "--config",
        args.config,
        "--universe-file",
        universe_file,
        "--trade-calendar-file",
        args.trade_calendar_file,
        "--workers",
        str(args.fundflow_workers),
        "--sleep",
        str(args.fundflow_sleep),
    ]
    if not args.fundflow_full:
        fundflow_cmd.append("--only-stale")
    if args.fundflow_require_columns.strip():
        fundflow_cmd.extend(["--require-columns", args.fundflow_require_columns.strip()])

    _run(manual_cmd)
    _run(fundflow_cmd)

    removed = _clear_post_update_disk_caches(cfg)
    _write_refresh_signal("bundle_update", {"ok": 1, "skip": 0, "empty": 0, "error": 0})
    print(
        f"[info] cache cleared: preselect={removed['preselect']} model_score_files={removed['model_scores']}"
    )

    if args.skip_prewarm:
        return

    provider = str((cfg.get("signals", {}) or {}).get("realtime_provider", "tencent") or "tencent").strip().lower()
    ok, fail = _post_refresh_prewarm(
        cfg,
        api_base=args.prewarm_api_base,
        provider=provider if provider in {"auto", "biying", "tencent", "sina", "netease"} else "tencent",
        timeout=args.prewarm_timeout,
    )
    print(f"[info] post-refresh prewarm: ok={ok} fail={fail} base={args.prewarm_api_base}")


if __name__ == "__main__":
    main()
