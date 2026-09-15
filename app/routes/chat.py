from fastapi import APIRouter, HTTPException

from app.schemas.chat_schema import ChatRequest, ChatResponse
from app.services.chat_service import (
    ChatServiceError,
    generate_answer,
)


router = APIRouter(
    prefix="/api",
    tags=["WAYVORA Chat"],
)


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if request.messages[-1].role != "user":
        raise HTTPException(
            status_code=422,
            detail="The last message must be a user question.",
        )

    for previous, current in zip(
        request.messages,
        request.messages[1:],
    ):
        if previous.role == current.role:
            raise HTTPException(
                status_code=422,
                detail="User and assistant messages must alternate.",
            )

    try:
        return generate_answer(request.messages)

    except ChatServiceError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=str(exc),
        ) from exc