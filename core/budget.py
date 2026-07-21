"""오늘 OpenRouter 사용 상한 가드. check()로 초과 여부, add(cost)로 누적.
날짜가 바뀌면 자동 리셋. 상한 초과 시 RuntimeError로 호출을 막는다."""
import json
import os
from datetime import date
from pathlib import Path

CAP = float(os.environ.get("IMG2FORM_DAILY_CAP", "20"))
_F = Path(os.environ.get("IMG2FORM_SPEND_FILE", "/tmp/img2form_test/spend_today.json"))


def _load():
    if _F.exists():
        d = json.loads(_F.read_text())
        if d.get("date") == date.today().isoformat():
            return float(d.get("usd", 0.0))
    return 0.0


def spent():
    return _load()


def check(estimate=0.0):
    """이번 호출 예상비용(estimate)까지 더해 상한 넘으면 즉시 중단."""
    s = _load()
    if s + estimate > CAP:
        raise RuntimeError(f"오늘 OpenRouter 사용 상한 ${CAP:.2f} 초과 방지 — 현재 ${s:.3f}, 예상 +${estimate:.3f}")
    return s


def add(cost):
    _F.parent.mkdir(parents=True, exist_ok=True)
    _F.write_text(json.dumps({"date": date.today().isoformat(), "usd": round(_load() + float(cost or 0), 6)}))
    return _load()
