import json
import logging
import os
import socket
import time

from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


logger = logging.getLogger(__name__)

LLAMA_BASE_URL = os.getenv(
    "LLAMA_BASE_URL",
    "http://127.0.0.1:8081",
).strip().rstrip("/")

LLAMA_API_KEY = os.getenv("LLAMA_API_KEY", "").strip()

# Settings for the deployed 1024-token model.
MAX_INPUT_TOKENS = 800
MAX_OUTPUT_TOKENS = 128
MAX_QUESTION_CHARACTERS = 400
TOTAL_TIMEOUT_SECONDS = 150

SYSTEM_PROMPT = """
You are WAYVORA, a career and education assistant.
Explain careers, courses, skills, and studying abroad.
Use plain language. Keep answers under 70 words.
Use relevant conversation details.
For qualification or licensing routes, ask the country if unknown.
Never invent institutions, exams, requirements, or statistics.
Never promise jobs, admission, or salaries.
If uncertain, say so. Do not repeat points.
""".strip()


class ChatServiceError(Exception):
    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


def post_json(path, payload, deadline):
    remaining = deadline - time.monotonic()

    if remaining <= 0:
        raise ChatServiceError(
            "The chatbot took too long. Please try again.",
            504,
        )

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    if LLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {LLAMA_API_KEY}"

    request = Request(
        url=f"{LLAMA_BASE_URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urlopen(request, timeout=remaining) as response:
            result = json.load(response)

        if not isinstance(result, dict):
            raise ValueError("Expected a JSON object")

        return result

    except HTTPError as exc:
        logger.warning(
            "AI endpoint %s returned HTTP %s",
            path,
            exc.code,
        )

        if exc.code in (401, 403):
            raise ChatServiceError(
                "The chatbot authentication failed. "
                "The administrator needs to check the AI API key.",
                503,
            ) from exc

        if exc.code in (429, 502, 503, 504):
            raise ChatServiceError(
                "The AI service is busy or waking up. "
                "Please try again shortly.",
                503,
            ) from exc

        raise ChatServiceError(
            "The AI service could not process this request. "
            "Try a shorter question or start a new chat.",
            502,
        ) from exc

    except (TimeoutError, socket.timeout) as exc:
        raise ChatServiceError(
            "The chatbot took too long. Please try again.",
            504,
        ) from exc

    except URLError as exc:
        raise ChatServiceError(
            "The AI service is temporarily unavailable. "
            "Please try again shortly.",
            503,
        ) from exc

    except (ValueError, UnicodeError) as exc:
        raise ChatServiceError(
            "The AI service returned an unreadable response.",
            502,
        ) from exc


def count_prompt_tokens(conversation, deadline):
    formatted = post_json(
        "/apply-template",
        {"messages": conversation},
        deadline,
    )

    prompt = formatted.get("prompt")

    if not isinstance(prompt, str):
        raise ChatServiceError(
            "The AI service could not prepare the conversation.",
            502,
        )

    tokenized = post_json(
        "/tokenize",
        {
            "content": prompt,
            "add_special": True,
            "parse_special": True,
        },
        deadline,
    )

    tokens = tokenized.get("tokens")

    if not isinstance(tokens, list):
        raise ChatServiceError(
            "The AI service could not check the conversation length.",
            502,
        )

    return len(tokens)


def generate_answer(messages):
    if not messages or messages[-1].role != "user":
        raise ChatServiceError(
            "Please enter a question.",
            422,
        )

    question = messages[-1].content.strip()

    if not question:
        raise ChatServiceError(
            "Please enter a question.",
            422,
        )

    if len(question) > MAX_QUESTION_CHARACTERS:
        raise ChatServiceError(
            "Please keep your question within 400 characters.",
            422,
        )

    deadline = time.monotonic() + TOTAL_TIMEOUT_SECONDS

    # Retain at most two previous question/answer pairs.
    selected = [
        {
            "role": message.role,
            "content": message.content,
        }
        for message in messages[-5:]
    ]

    while selected and selected[0]["role"] != "user":
        selected.pop(0)

    selected[-1]["content"] = question

    # Drop the oldest complete pair if the conversation is too long.
    while True:
        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *selected,
        ]

        token_count = count_prompt_tokens(
            conversation,
            deadline,
        )

        if token_count <= MAX_INPUT_TOKENS:
            break

        if len(selected) <= 1:
            raise ChatServiceError(
                "This question exceeds the chatbot's context limit. "
                "Please shorten it.",
                422,
            )

        selected = selected[2:]

    result = post_json(
        "/v1/chat/completions",
        {
            "model": "wayvora-chat",
            "messages": conversation,
            "temperature": 0.3,
            "repeat_penalty": 1.1,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "stream": False,
        },
        deadline,
    )

    try:
        choice = result["choices"][0]
        answer = choice["message"]["content"]

        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Empty answer")

        return {
            "answer": answer.strip(),
            "truncated": choice.get("finish_reason") == "length",
        }

    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ChatServiceError(
            "The chatbot returned no usable answer. Please try again.",
            502,
        ) from exc
