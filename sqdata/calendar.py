from __future__ import annotations

import datetime as dt


def resolve_trade_date(date_str: str) -> str:
    """
    v1: crude trade date resolver.
    - If empty: use today
    - If weekend: roll back to Friday
    """
    if date_str:
        return date_str

    d = dt.date.today()
    # 5=Saturday, 6=Sunday
    if d.weekday() == 5:
        d = d - dt.timedelta(days=1)
    elif d.weekday() == 6:
        d = d - dt.timedelta(days=2)
    return d.isoformat()