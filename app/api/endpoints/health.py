import time
from fastapi import APIRouter
from app.models.responses import HealthResponse
from app.config.settings import settings

router = APIRouter(tags=["health"])
START_TIME = time.time()

@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
async def health_check() -> HealthResponse:
    return HealthResponse(
        success=True,
        message="Service healthy",
        version=settings.app_version,
        environment=settings.environment,
        uptime_seconds=int(time.time() - START_TIME),
        database_status="ok"
    )
