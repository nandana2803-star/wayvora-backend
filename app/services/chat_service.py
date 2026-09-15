import json
import logging
import socket

from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


logger = logging.getLogger(__name__)

CHAT_URL = "http://127.0.0.1:8081/v1/chat/completions"

SYSTEM_PROMPT = """
You are WAYVORA Assistant, a career and education guide.
Help users explore careers, courses, qualifications, skills,
interviews, and career changes.

ACCURACY
- Prefer an honest, limited answer over an invented explanation.
- Separate general guidance from formal requirements.
- Never invent institutions, professional bodies, qualifications,
  acronyms, sources, links, statistics, or personal experience.
- Do not describe a common route as the only mandatory route.
- If uncertain about a detail, omit it or state the uncertainty.
- Never promise admission, employment, salary, or career success.
- Do not claim you searched the internet or verified information.

UNDERSTAND THE QUESTION
- Answer what the user actually asked.
- Use relevant details from the conversation.
- Never assume the user's country, age, education, budget,
  interests, academic performance, or work experience.
- For qualification, licensing, or admission procedures, first
  establish the country if it is not already known.
- For a personalised pathway, ask about current education when
  it would change the answer.
- Ask one focused clarifying question at a time.
- General questions such as "What does a civil engineer do?"
  can be answered directly without asking for a country.
- If an abbreviation has several meanings, ask what it means.

CAREER AND COURSE GUIDANCE
- Consider all relevant fields, not just technology.
- Suggest options based on the user's stated interests.
- Explain why an option may fit and mention relevant tradeoffs.
- Distinguish degrees, diplomas, certificates, professional
  qualifications, and licences.
- Do not assume a bachelor's degree is required for every route.
- For comparisons, compare the same factors for both options.
- For roadmaps, give a short sequence of distinct, practical steps.
- If asked for the "best" option, explain that the choice depends
  on the user's goals and constraints.

CURRENT OR LOCATION-SPECIFIC INFORMATION
- Fees, salaries, deadlines, rankings, immigration rules,
  accreditation, and admission requirements can change.
- Do not provide exact current figures or rules as verified facts.
- Explain what the user should check with the relevant official
  institution or professional body.
- Do not invent the name or website of that body.

CONVERSATION AND WRITING
- Use plain language and a friendly, professional tone.
- Default to 80-150 words and no more than five bullet points.
- Each point must add new information.
- Do not repeat qualifications, steps, warnings, or conclusions.
- Finish the answer once the question has been addressed.
- For follow-up questions, answer the new question without
  repeating the whole previous response.
- If an earlier answer was wrong, acknowledge and correct it.
  Do not treat your previous answers as verified evidence.
- If asked to fabricate facts, qualifications, or work experience,
  decline that part and offer an honest alternative.
- For unrelated requests, briefly explain your career and
  education focus.

EXAMPLES
User: How can I become a chartered accountant?
Assistant: Which country do you want to qualify in? The
qualification route depends on the country.

User: What does a civil engineer do?
Assistant: A civil engineer helps design, build, and maintain
infrastructure such as roads, bridges, buildings, and water
systems. The work can include planning, calculations, site
supervision, and checking safety and quality.

User: Which course guarantees a high-paying job?
Assistant: No course guarantees a job or salary. What subjects
do you enjoy, and what is your current level of education?

STUDY ABROAD
- Studying abroad is a supported career and education topic.
- Use the conversation to identify the field of study.
- If the user has been discussing doctors or medicine and asks
  about studying abroad, interpret it as studying medicine abroad.
- Ask which destination country interests them.
- If they have not chosen one, explain general comparison factors:
  teaching language, costs, admission requirements, recognition
  of the qualification, and requirements to practise after graduation.
- Do not invent country-specific rules or guarantee recognition.
- Educational questions about becoming a doctor are different
  from requests for medical diagnosis or treatment.
- Do not refuse a study-abroad question merely because it is broad.

Example:
User: I want to become a doctor.
Assistant: What would you like to know about studying medicine?
User: What about abroad study?
Assistant: Are you interested in studying medicine abroad?
Which country would you like to study in?

""".strip()


class ChatServiceError(Exception):
    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


def generate_answer(messages):
    conversation = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    # Keep the latest complete turns within a modest input budget.
    selected = []
    characters = 0

    for message in reversed(messages):
        if characters + len(message.content) > 4000:
            break

        selected.append(
            {
                "role": message.role,
                "content": message.content,
            }
        )

        characters += len(message.content)

    selected.reverse()

    # A shortened conversation should start with a user message.
    while selected and selected[0]["role"] != "user":
        selected.pop(0)

    conversation.extend(selected)

    payload = {
        "model": "wayvora-chat",
        "messages": conversation,
        "temperature": 0.3,
        "repeat_penalty": 1.15,
        "repeat_last_n": 256,
        "max_tokens": 300,
        "stream": False,
    }

    request = Request(
        CHAT_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=180) as response:
            data = json.load(response)

    except HTTPError as exc:
        logger.warning("Chat model returned HTTP %s", exc.code)

        if exc.code in (429, 503):
            raise ChatServiceError(
                "The chatbot is busy or still loading. Please try again.",
                503,
            ) from exc

        raise ChatServiceError(
            "The model could not process this conversation. "
            "Try a shorter question or start a new chat.",
            502,
        ) from exc

    except (TimeoutError, socket.timeout) as exc:
        raise ChatServiceError(
            "The chatbot took too long to respond. Please try again.",
            504,
        ) from exc

    except URLError as exc:
        raise ChatServiceError(
            "Cannot connect to the chatbot. "
            "Make sure start-chatbot.ps1 is running on port 8081.",
            503,
        ) from exc

    except (ValueError, UnicodeError) as exc:
        raise ChatServiceError(
            "The chatbot returned an unreadable response. Please try again.",
            502,
        ) from exc

    try:
        choice = data["choices"][0]
        answer = choice["message"]["content"]
        finish_reason = choice.get("finish_reason")

        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Empty answer")

    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ChatServiceError(
            "The chatbot returned no usable answer. Please try again.",
            502,
        ) from exc

    return {
        "answer": answer.strip(),
        "truncated": finish_reason == "length",
    }