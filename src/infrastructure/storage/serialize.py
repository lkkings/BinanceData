"""UnifiedMarketData 的 JSON 序列化 / 反序列化"""
import json
from dataclasses import fields
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from ...domain import UnifiedMarketData


_FIELDS = fields(UnifiedMarketData)
_DECIMAL_NAMES = {f.name for f in _FIELDS if "Decimal" in str(f.type)}

_CST = timezone(timedelta(hours=8))


def _encode(value: Any) -> Any:
    """存储编码：datetime→epoch s, Decimal→str"""
    if value is None:
        return None
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return int(value.timestamp())
    return value


def _encode_ws(value: Any) -> Any:
    """WS 推送编码：datetime→UTC+8 字符串, Decimal→str"""
    if value is None:
        return None
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.astimezone(_CST).strftime("%Y-%m-%d %H:%M:%S")
    return value


def to_json(record: UnifiedMarketData) -> str:
    """存储用 JSON（timestamp=epoch s）"""
    return json.dumps(
        {f.name: _encode(getattr(record, f.name)) for f in _FIELDS},
        separators=(",", ":"),
    )


def to_payload_dict(record: UnifiedMarketData) -> dict[str, Any]:
    """WS 推送用 dict（timestamp=UTC+8 字符串）"""
    return {f.name: _encode_ws(getattr(record, f.name)) for f in _FIELDS}


def epoch_seconds(record: UnifiedMarketData) -> int:
    return int(record.timestamp.timestamp())


def from_json(blob: str | bytes) -> UnifiedMarketData:
    if isinstance(blob, bytes):
        blob = blob.decode("utf-8")
    payload = json.loads(blob)
    kwargs: dict[str, Any] = {}
    for f in _FIELDS:
        raw = payload.get(f.name)
        if raw is None:
            kwargs[f.name] = None
            continue
        if f.name == "timestamp":
            kwargs[f.name] = datetime.fromtimestamp(int(raw), tz=timezone.utc)
        elif f.name in _DECIMAL_NAMES:
            kwargs[f.name] = Decimal(str(raw))
        else:
            kwargs[f.name] = raw
    return UnifiedMarketData(**kwargs)
