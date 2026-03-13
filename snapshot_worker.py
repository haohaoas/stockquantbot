from __future__ import annotations

import time

import api


def main() -> None:
    api._apply_external_cache_bust_if_needed()
    api._start_snapshot_scheduler()
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
