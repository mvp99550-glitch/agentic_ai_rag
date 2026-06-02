"""
Request context middleware — extracts and validates tenant/user/session from every request.

For this pilot, IDs come from request headers or query params.
In production behind AWS API Gateway, tenant_id will come from the JWT/authorizer.
"""

from fastapi import Header, HTTPException
from typing import Optional
import uuid


class RequestContext:
    def __init__(self, tenant_id: str, user_id: str, session_id: str):
        self.tenant_id  = tenant_id
        self.user_id    = user_id
        self.session_id = session_id

    @property
    def redis_prefix(self) -> str:
        """Namespaced Redis key prefix — isolates data per tenant/user/session."""
        return f"{self.tenant_id}:{self.user_id}:{self.session_id}"

    @property
    def log_prefix(self) -> str:
        return f"[{self.tenant_id}/{self.user_id}]"


def get_context(
    x_tenant_id:  Optional[str] = Header(default="default_tenant"),
    x_user_id:    Optional[str] = Header(default="default_user"),
    x_session_id: Optional[str] = Header(default=None),
) -> RequestContext:
    """
    FastAPI dependency — injects context into any route.

    Headers expected:
        X-Tenant-Id   (default: 'default_tenant')
        X-User-Id     (default: 'default_user')
        X-Session-Id  (auto-generated UUID if missing)

    In production: X-Tenant-Id is set by AWS API Gateway from the JWT claims.
    """
    if not x_tenant_id or not x_tenant_id.strip():
        raise HTTPException(status_code=400, detail="X-Tenant-Id header is required")
    if not x_user_id or not x_user_id.strip():
        raise HTTPException(status_code=400, detail="X-User-Id header is required")

    session_id = x_session_id or str(uuid.uuid4())
    return RequestContext(
        tenant_id=x_tenant_id.strip(),
        user_id=x_user_id.strip(),
        session_id=session_id,
    )
