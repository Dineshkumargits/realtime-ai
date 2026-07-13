"""Text filters applied to LLM output before TTS synthesis.

Small/medium models under a long, complex persona prompt sometimes write
stage directions into the dialogue itself (e.g. "(Phone rings, I answer)",
"(Slight chuckle)") -- harmless as text, but Kokoro will read these out loud
verbatim as spoken words since nothing else in the pipeline strips them.

Hybrid thinking models (Qwen3) can also leak an empty `<think></think>`
wrapper even with the `/no_think` prefix set -- same problem, TTS would
otherwise read the literal tag text out loud.
"""

from __future__ import annotations

import re

from pipecat.utils.text.base_text_filter import BaseTextFilter

_LONG_DIGIT_RUN_RE = re.compile(r"\d{7,}")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_UNCLOSED_THINK_RE = re.compile(r"<think>.*", re.DOTALL)
_STRAY_THINK_TAG_RE = re.compile(r"</?think>")
_PAREN_RE = re.compile(r"\([^)]*\)")
_ASTERISK_ACTION_RE = re.compile(r"\*[^*]+\*")
_WHITESPACE_RE = re.compile(r"[ \t]{2,}")


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> wrappers (paired, unclosed, or stray) so the
    oai-events transcript the client displays matches what TTS actually says,
    not vLLM's raw hybrid-thinking-model output.
    """
    text = _THINK_RE.sub("", text)
    text = _UNCLOSED_THINK_RE.sub("", text)
    text = _STRAY_THINK_TAG_RE.sub("", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


class StageDirectionTextFilter(BaseTextFilter):
    """Strips parenthetical/asterisk stage directions and <think> tags from
    LLM output.

    E.g. "(Phone rings, I answer) Hello?" -> "Hello?"
    E.g. "<think></think>Hello?" -> "Hello?"
    """

    async def filter(self, text: str) -> str:
        text = strip_think_tags(text)
        text = _PAREN_RE.sub("", text)
        text = _ASTERISK_ACTION_RE.sub("", text)
        text = _LONG_DIGIT_RUN_RE.sub(lambda m: " ".join(m.group()), text)
        return _WHITESPACE_RE.sub(" ", text).strip()
