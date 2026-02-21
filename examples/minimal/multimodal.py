from __future__ import annotations

import asyncio
import base64
import sys
import tempfile
import urllib.request
from pathlib import Path

EXAMPLES_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from common import run_case
from fabrix import Agent
from fabrix.messages import ImageMessage, TextMessage

PYTHON_LOGO_URL = "https://raw.githubusercontent.com/github/explore/main/topics/python/python.png"
JAVASCRIPT_LOGO_URL = (
    "https://raw.githubusercontent.com/github/explore/main/topics/javascript/javascript.png"
)

# 1x1 PNG fallback in case external image download fails.
_FALLBACK_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
)


def _load_image_bytes(url: str) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=10) as response:  # noqa: S310
            return response.read()
    except Exception:
        return base64.b64decode(_FALLBACK_PNG_BASE64)


async def _run_case(agent: Agent, title: str, messages: list[TextMessage | ImageMessage]) -> None:
    await run_case(agent, title=title, messages=messages)


async def main() -> None:
    agent = Agent(
        instructions=(
            "You are a visual analysis assistant. "
            "Use image evidence, state uncertainty clearly, and keep outputs compact."
        ),
        model="gpt-5.3-codex",
    )

    python_logo_bytes = _load_image_bytes(PYTHON_LOGO_URL)

    with tempfile.TemporaryDirectory(prefix="fabrix-mm-") as temp_dir:
        local_image_path = Path(temp_dir) / "python_logo.png"
        local_image_path.write_bytes(python_logo_bytes)

        await _run_case(
            agent,
            "Case 1: URL image summary",
            [
                TextMessage(text="Summarize what this image likely represents."),
                ImageMessage(image=PYTHON_LOGO_URL),
                TextMessage(text="Return 3 bullet points."),
            ],
        )

        await _run_case(
            agent,
            "Case 2: Local path image description",
            [
                TextMessage(text="Describe the shape, color palette, and probable category."),
                ImageMessage(image=local_image_path),
                TextMessage(text="Keep it to 4 short lines."),
            ],
        )

        await _run_case(
            agent,
            "Case 3: Raw bytes image inspection",
            [
                TextMessage(text="You will receive image bytes. Analyze what you can infer."),
                ImageMessage(image=python_logo_bytes),
                TextMessage(text="Include one explicit uncertainty statement."),
            ],
        )

        await _run_case(
            agent,
            "Case 4: Two-image comparison",
            [
                TextMessage(text="Compare two logos and list key visual differences."),
                ImageMessage(image=PYTHON_LOGO_URL, text="First image"),
                ImageMessage(image=JAVASCRIPT_LOGO_URL, text="Second image"),
                TextMessage(text="Output: style, dominant colors, and likely language/theme."),
            ],
        )


if __name__ == "__main__":
    asyncio.run(main())
