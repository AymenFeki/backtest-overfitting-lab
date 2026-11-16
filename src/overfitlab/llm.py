"""The one place where a language model is called.

`complete` sends a conversation to the provider named in the config (a local
model through Ollama, or Claude through the Anthropic API) and returns the
reply. Three rules:

* Every reply is stored on disk under a hash of the full request. Asking the
  same thing again reads the stored reply, so nothing is computed or paid for
  twice and experiments can be re-scored without any model call.
* A new call is only made with `live=True`. Without it a missing reply is an
  error, so rerunning the experiments can never start model calls by accident.
* API calls keep a running total in dollars. A call that could take the total
  past the cap is refused before it is sent.
"""

import hashlib
import json
import os
import time
import urllib.request
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path

from . import config as cfg

CACHE_DIR = Path("cache/llm")


@dataclass(frozen=True)
class Model:
    provider: str  # "ollama" or "anthropic"
    name: str


@dataclass(frozen=True)
class Reply:
    text: str
    input_tokens: int  # all prompt tokens, cached or not
    output_tokens: int
    seconds: float  # time the model spent generating
    cost_usd: float = 0.0
    stop: str = "stop"


class CacheMiss(RuntimeError):
    """A reply is not stored and live calls are off."""


class BudgetExceeded(RuntimeError):
    """The next API call could take spending past the cap."""


def _path(model: Model) -> Path:
    return CACHE_DIR / f"{model.provider}__{model.name.replace(':', '_')}.jsonl"


@cache
def _stored(model: Model) -> dict[str, Reply]:
    """All stored replies of one model, read once per process."""
    path = _path(model)
    if not path.exists():
        return {}
    lines = (json.loads(line) for line in path.read_text().splitlines())
    return {line.pop("key"): Reply(**line) for line in lines}


def spent_usd(model: Model) -> float:
    """Total cost of every stored reply of this model."""
    return sum(reply.cost_usd for reply in _stored(model).values())


def complete(
    model: Model, system: str, messages: list[dict], seed: int, live: bool = False
) -> Reply:
    """Reply to a conversation; `messages` alternate user and assistant turns."""
    request = {
        "model": asdict(model),
        "system": system,
        "messages": messages,
        "seed": seed,
        "temperature": cfg.LLM_TEMPERATURE,
        "max_tokens": cfg.LLM_MAX_TOKENS,
    }
    key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    stored = _stored(model)
    if key in stored:
        return stored[key]
    if not live:
        raise CacheMiss(f"no stored reply from {model.name}; rerun with --live")
    call = _ollama if model.provider == "ollama" else _anthropic
    reply = call(model, system, messages, seed)
    stored[key] = reply
    path = _path(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as file:
        file.write(json.dumps({"key": key} | asdict(reply)) + "\n")
    return reply


# --- local model through Ollama ----------------------------------------------
def _ollama_post(endpoint: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body else None
    request = urllib.request.Request(f"{cfg.OLLAMA_URL}/api/{endpoint}", data=data)
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


@cache
def _installed(name: str) -> dict:
    """Ollama's record of an installed model: digest, capabilities, size."""
    models = {m["name"]: m for m in _ollama_post("tags")["models"]}
    return models[name]


def _ollama(model: Model, system: str, messages: list[dict], seed: int) -> Reply:
    record = _installed(model.name)
    pinned = cfg.OLLAMA_DIGESTS.get(model.name)
    if pinned and not record["digest"].startswith(pinned):
        raise RuntimeError(f"{model.name} is not the pinned version {pinned}")
    body = {
        "model": model.name,
        "messages": [{"role": "system", "content": system}, *messages],
        "stream": False,
        "keep_alive": "10m",
        "options": {
            "seed": seed,
            "temperature": cfg.LLM_TEMPERATURE,
            "num_predict": cfg.LLM_MAX_TOKENS,
            "num_ctx": cfg.OLLAMA_CONTEXT,
        },
    }
    if "thinking" in record.get("capabilities", []):
        body["think"] = False  # plain replies from every model, as with the API
    answer = _ollama_post("chat", body)
    if answer.get("prompt_eval_count", 0) + cfg.LLM_MAX_TOKENS > cfg.OLLAMA_CONTEXT:
        raise RuntimeError("the prompt no longer fits the context; it would be cut")
    return Reply(
        text=answer["message"]["content"],
        input_tokens=answer.get("prompt_eval_count", 0),
        output_tokens=answer.get("eval_count", 0),
        seconds=answer.get("total_duration", 0) / 1e9,
        stop=answer.get("done_reason", "stop"),
    )


# --- Claude through the Anthropic API ----------------------------------------
def _anthropic(model: Model, system: str, messages: list[dict], seed: int) -> Reply:
    """One API call. The key is read from the ANTHROPIC_API_KEY environment
    variable by the SDK. This model takes no seed and no temperature; `seed`
    only tells repeated runs apart in the cache."""
    price_in, price_out = cfg.API_PRICE_PER_MTOK[model.name]
    # Worst case for the pre-call check: every prompt character is a token
    # written to the prompt cache (1.25x), and the reply uses all its tokens.
    characters = len(system) + sum(len(m["content"]) for m in messages)
    worst = (characters * 1.25 * price_in + cfg.LLM_MAX_TOKENS * price_out) / 1e6
    if spent_usd(model) + worst > cfg.API_BUDGET_USD:
        raise BudgetExceeded(
            f"spent ${spent_usd(model):.4f} of ${cfg.API_BUDGET_USD:.2f}; "
            f"the next call could cost ${worst:.4f}"
        )
    if "ANTHROPIC_API_KEY" not in os.environ:
        raise RuntimeError("set ANTHROPIC_API_KEY in your shell first")
    import anthropic  # imported here so the local-model path does not need it

    start = time.time()
    answer = anthropic.Anthropic().messages.create(
        model=model.name,
        max_tokens=cfg.LLM_MAX_TOKENS,
        system=system,
        messages=messages,
        thinking={"type": "disabled"},  # plain text in, plain text out
        cache_control={"type": "ephemeral"},  # reuse the growing conversation
    )
    usage = answer.usage
    written = usage.cache_creation_input_tokens or 0
    read = usage.cache_read_input_tokens or 0
    cost = (
        (usage.input_tokens + 1.25 * written + 0.1 * read) * price_in
        + usage.output_tokens * price_out
    ) / 1e6
    return Reply(
        text="".join(block.text for block in answer.content if block.type == "text"),
        input_tokens=usage.input_tokens + written + read,
        output_tokens=usage.output_tokens,
        seconds=time.time() - start,
        cost_usd=cost,
        stop=answer.stop_reason or "stop",
    )
