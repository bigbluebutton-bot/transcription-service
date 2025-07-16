from pydantic import BaseModel
from typing import Literal

class APIendpointResponse(BaseModel):
    method: str
    path: str

class APIHealthResponse(BaseModel):
    status: Literal["starting", "running", "stopping", "stopped"]
    version: str