"""FastAPI 应用工厂"""
from __future__ import annotations

from fastapi import FastAPI, Request

from .deps import Container
from .lifespan import lifespan
from .range_query import router as range_router
from .websocket import router as ws_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Binance Realtime Market Data",
        version="0.2.0",
        description="多交易对实时分钟级聚合 + WebSocket 推送服务",
        lifespan=lifespan,
    )

    app.include_router(ws_router)
    app.include_router(range_router)

    @app.get("/healthz")
    async def healthz(request: Request) -> dict:
        c: Container = request.app.state.container
        return {
            "status": "ok",
            "symbols": c.pipeline.symbols,
            "subscribers": c.pubsub.subscriber_count,
            "retry_jobs": len(c.retry_schedulers),
        }

    @app.get("/symbols")
    async def list_symbols(request: Request) -> dict:
        c: Container = request.app.state.container
        return {"symbols": c.pipeline.symbols, "channels": ["minute"]}

    return app
