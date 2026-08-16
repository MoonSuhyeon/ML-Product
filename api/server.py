"""FastAPI — 수요 예측 조회.

    GET /forecast              예측 (as_of 를 함께 준다)
    GET /forecast/low-demand   수요가 낮은 숙소·날짜 → 콘텐츠 생성으로 넘어가는 지점
    GET /forecast/segments     지역별 오차 — 평균 뒤에 가려지는 것
    GET /metrics               모델 비교 (측정해 둔 결과)

배치 파이프라인 위에 얹은 얇은 조회 층이다. 학습은 여기서 하지 않는다 —
``scripts/run_baseline.py`` 가 하고, 이 API 는 그 결과와 예측을 보여준다.
"""
from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from api.routers import forecast, metrics
from api.state import runtime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("forecast_api")

app = FastAPI(
    title="Demand Forecast API",
    description="as-of 피처로 만든 숙박 수요 예측을 조회한다.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    logger.info("%s %s → %s (%.0fms)", request.method, request.url.path,
                response.status_code, (time.time() - t0) * 1000)
    return response


app.include_router(forecast.router)
app.include_router(metrics.router)


@app.get("/health", tags=["system"])
def health() -> dict:
    return {"status": "ok", "panel_ready": runtime._panel is not None}
