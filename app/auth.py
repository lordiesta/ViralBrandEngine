from __future__ import annotations

import secrets

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from app.config import get_settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)


def _valid_key(candidate: str | None, expected: str) -> bool:
    return bool(candidate and expected and secrets.compare_digest(candidate, expected))


def require_api_access(
    api_key: str | None = Security(api_key_header),
    bearer: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
) -> None:
    settings = get_settings()
    candidate = api_key or (bearer.credentials if bearer else None)
    if settings.API_KEY and _valid_key(candidate, settings.API_KEY):
        return
    if settings.ADMIN_API_KEY and _valid_key(candidate, settings.ADMIN_API_KEY):
        return
    if not settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API access is not configured. Set API_KEY before exposing this service.",
        )
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key.")


def require_admin_access(
    api_key: str | None = Security(api_key_header),
    bearer: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
) -> None:
    settings = get_settings()
    candidate = api_key or (bearer.credentials if bearer else None)
    if settings.ADMIN_API_KEY and _valid_key(candidate, settings.ADMIN_API_KEY):
        return
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Administrator access required.")
