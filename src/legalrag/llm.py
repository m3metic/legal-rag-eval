"""One small interface over the model providers, with a disk cache.

Models are named "<provider>:<model id>", for example "anthropic:claude-haiku-5-5"
or "openai:gpt-6.1-sol", so an experiment config can switch provider by changing
one string. API keys come from ANTHROPIC_API_KEY and OPENAI_API_KEY.

Every response is cached under .cache/llm, keyed by model, system prompt and
prompt. Rerunning an experiment (to recompute metrics, or after a crash) costs
nothing for calls already made, and results stay reproducible.

"fake:<anything>" is a stand-in model for tests: it returns whatever the
`responder` function gives it and never touches the network.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CACHE = Path(".cache/llm")


@dataclass
class Usage:
    calls: int = 0
    cached_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class LLM:
    spec: str  # "<provider>:<model id>"
    max_tokens: int = 1024
    cache_dir: Path | None = DEFAULT_CACHE
    responder: Callable[[str, str], str] | None = None  # only for provider "fake"
    usage: Usage = field(default_factory=Usage)

    def __post_init__(self):
        if ":" not in self.spec:
            raise ValueError(f"Model must be '<provider>:<model id>', got {self.spec!r}")
        self.provider, self.model = self.spec.split(":", 1)
        if self.provider not in ("anthropic", "openai", "fake"):
            raise ValueError(f"Unknown provider {self.provider!r}; use anthropic, openai or fake")
        self._client = None

    @property
    def family(self) -> str:
        """Model family, used to warn when the grader shares the writer's family."""
        return self.provider

    def complete(self, system: str, prompt: str) -> str:
        key = hashlib.sha256(json.dumps([self.spec, self.max_tokens, system, prompt]).encode()).hexdigest()
        path = self.cache_dir / f"{key}.json" if self.cache_dir else None
        if path and path.exists():
            self.usage.cached_calls += 1
            return json.loads(path.read_text(encoding="utf-8"))["text"]

        text, n_in, n_out = self._call(system, prompt)
        self.usage.calls += 1
        self.usage.input_tokens += n_in
        self.usage.output_tokens += n_out
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"model": self.spec, "text": text}), encoding="utf-8")
        return text

    def _call(self, system: str, prompt: str) -> tuple[str, int, int]:
        if self.provider == "fake":
            if self.responder is None:
                raise ValueError("A fake model needs a responder function")
            return self.responder(system, prompt), 0, 0

        if self.provider == "anthropic":
            if self._client is None:
                import anthropic

                self._client = anthropic.Anthropic(max_retries=5)
            r = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
            return text, r.usage.input_tokens, r.usage.output_tokens

        # openai
        if self._client is None:
            import openai

            self._client = openai.OpenAI(max_retries=5)
        r = self._client.responses.create(
            model=self.model,
            instructions=system,
            input=prompt,
            max_output_tokens=self.max_tokens,
        )
        return r.output_text, r.usage.input_tokens, r.usage.output_tokens
