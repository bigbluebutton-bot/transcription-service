from pydantic import BaseModel
from app.core.config import SystemStatusType

class APIendpointResponse(BaseModel):
    method: str
    path: str

class APIHealthResponse(BaseModel):
    status: SystemStatusType
    version: str