"""共享 pytest fixtures"""
import asyncio
import os
import sys
from pathlib import Path

import pytest

# 让 import src.* 可用
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 测试期间默认走 fakeredis
os.environ.setdefault("REDIS_URL", "fake://")
os.environ.setdefault("HISTORY_BACKFILL_DAYS", "0")  # 默认跳过回填，避免外网
os.environ.setdefault("LOG_LEVEL", "WARNING")


@pytest.fixture
def event_loop_policy():
    return asyncio.DefaultEventLoopPolicy()
