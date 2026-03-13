from __future__ import annotations

import time

import api


def main() -> None:
    api._apply_external_cache_bust_if_needed()
    result = api._run_startup_snapshot_prewarm()
    print(f"startup prewarm: ok={result.get('ok', 0)} skip={result.get('skip', 0)} fail={result.get('fail', 0)}")
    api._start_snapshot_scheduler()
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
