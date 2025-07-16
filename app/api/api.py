from fastapi import APIRouter

from app.api.routes import books

api_router = APIRouter()
api_router.include_router(books.router, prefix="/books", tags=["books"])
