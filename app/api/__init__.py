"""
API package initialization and router setup.
"""

from fastapi import APIRouter
from app.api.endpoints import chat, health   # ← ambos son APIRouter

# Main API router
api_router = APIRouter()

# Incluir sub-routers
api_router.include_router(chat)
api_router.include_router(health)

__all__ = ["api_router"]