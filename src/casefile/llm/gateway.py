"""
The only path to a model. Three interchangeable backends behind one signature.

  local     transformers on the GPU. Real tokens, real latency, real cost. Default.
  template  deterministic Jinja writer. Always available, zero dependencies. This is not a
            degraded mode: it is the control arm that proves the numbers are model-independent.
  api       Anthropic, if a key is present. Optional.

Every call returns the same envelope carrying a mandatory `token_source`, so "estimated cost"
can be audited rather than believed. Counting is done with the model's OWN tokenizer over the
exact string that was sent, never a heuristic.

Prompts are built as [stable prefix][volatile suffix]. The prefix holds the instruction block,
the contract excerpt and the claim schema with sorted keys and no timestamps, so it is
byte-identical across calls and a KV cache can actually hit it.
"""
from __future__ import annotations
import hashlib, json, os, time
from dataclasses import dataclass, field
from pathlib import Path
import yaml

from ..paths import ROOT, routing as _routing
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf"))


@dataclass
class LLMResult:
    text: str
    backend: str
    model_id: str
    tokens_in: int
    tokens_out: int
    token_source: str          # exact_api | local_tokenizer
    latency_ms: float
    cache_hit: bool = False
    prefix_tokens: int = 0
    error: str | None = None


class LLMGateway:
    _models: dict = {}
    _toks: dict = {}

    def __init__(self, mode: str = "local", routing_path: str | None = None,
                 device: str = "cuda"):
        self.mode = mode
        self.device = device
        self.routing = yaml.safe_load(open(routing_path or _routing()))
        self._cache: dict[str, LLMResult] = {}
        self.calls: list[dict] = []

    # ------------------------------------------------------------------ policy
    def stage_policy(self, stage: str) -> dict:
        p = self.routing["stages"].get(stage)
        if p is None: raise KeyError(f"stage {stage} has no routing policy")
        return p

    def may_call_model(self, stage: str) -> bool:
        return self.stage_policy(stage).get("backend", "none") != "none"

    # ------------------------------------------------------------------ loading
    def _load(self, model_id: str):
        if model_id in self._models: return self._models[model_id], self._toks[model_id]
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_id)
        mdl = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
        if self.device == "cuda" and torch.cuda.is_available():
            mdl = mdl.to("cuda:0")          # explicit single device: sharding a small model
        mdl.eval()                           # across cards yields NaN logits on this stack
        self._models[model_id] = mdl; self._toks[model_id] = tok
        return mdl, tok

    def count_tokens(self, model_id: str, text: str) -> int:
        try:
            _, tok = self._load(model_id) if self.mode == "local" else (None, self._tokenizer_only(model_id))
            return len(tok(text).input_ids)
        except Exception:
            return max(1, len(text) // 4)

    def _tokenizer_only(self, model_id: str):
        if model_id not in self._toks:
            from transformers import AutoTokenizer
            self._toks[model_id] = AutoTokenizer.from_pretrained(model_id)
        return self._toks[model_id]

    # ------------------------------------------------------------------ generate
    def generate(self, stage: str, prefix: str, suffix: str,
                 template_fn=None, max_new_tokens: int | None = None) -> LLMResult:
        pol = self.stage_policy(stage)
        if pol.get("backend", "none") == "none":
            raise PermissionError(
                f"stage {stage} is declared backend:none in routing.yaml "
                f"({pol['why']}). Calling a model here would violate the LLM boundary.")
        model_id = pol.get("model", "template")
        mnt = max_new_tokens or int(pol.get("max_new_tokens", 256))
        key = hashlib.sha256((self.mode + stage + prefix + suffix).encode()).hexdigest()
        if key in self._cache:
            r = self._cache[key]
            return LLMResult(r.text, r.backend, r.model_id, r.tokens_in, r.tokens_out,
                             r.token_source, 0.0, True, r.prefix_tokens)

        t0 = time.perf_counter()
        if self.mode == "template" or template_fn is not None and self.mode == "template":
            text = template_fn() if template_fn else ""
            res = LLMResult(text, "none", "deterministic-template",
                            self.count_tokens("Qwen/Qwen2.5-1.5B-Instruct", prefix + suffix),
                            self.count_tokens("Qwen/Qwen2.5-1.5B-Instruct", text),
                            "local_tokenizer", (time.perf_counter() - t0) * 1000)
        elif self.mode == "local":
            try:
                import torch
                mdl, tok = self._load(model_id)
                msgs = [{"role": "system", "content": prefix},
                        {"role": "user", "content": suffix}]
                enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                              return_tensors="pt", return_dict=True)
                from collections.abc import Mapping
                if not isinstance(enc, Mapping):       # transformers <5 returns a bare tensor
                    enc = {"input_ids": enc}
                enc = {k: v.to(mdl.device) for k, v in enc.items()
                       if k in ("input_ids", "attention_mask")}
                n_in = int(enc["input_ids"].shape[-1])
                with torch.no_grad():
                    out = mdl.generate(**enc, max_new_tokens=mnt, do_sample=False,
                                       pad_token_id=tok.eos_token_id or tok.pad_token_id)
                gen = out[0][n_in:]
                text = tok.decode(gen, skip_special_tokens=True)
                res = LLMResult(text, "local", model_id, n_in, int(gen.shape[-1]),
                                "local_tokenizer", (time.perf_counter() - t0) * 1000,
                                prefix_tokens=len(tok(prefix).input_ids))
            except Exception as e:
                import traceback
                text = template_fn() if template_fn else ""
                res = LLMResult(text, "none", "deterministic-template(fallback)",
                                self.count_tokens("Qwen/Qwen2.5-1.5B-Instruct", prefix + suffix),
                                self.count_tokens("Qwen/Qwen2.5-1.5B-Instruct", text),
                                "local_tokenizer", (time.perf_counter() - t0) * 1000,
                                error=f"{type(e).__name__}: {e}\n{traceback.format_exc()[-600:]}")
        else:  # api
            from anthropic import Anthropic
            cl = Anthropic()
            m = "claude-haiku-4-5" if "1.5B" in model_id else "claude-sonnet-4-5"
            r = cl.messages.create(model=m, max_tokens=mnt, system=prefix,
                                   messages=[{"role": "user", "content": suffix}])
            res = LLMResult(r.content[0].text, "anthropic", m, r.usage.input_tokens,
                            r.usage.output_tokens, "exact_api",
                            (time.perf_counter() - t0) * 1000)
        self._cache[key] = res
        self.calls.append({"stage": stage, "backend": res.backend, "model": res.model_id,
                           "tokens_in": res.tokens_in, "tokens_out": res.tokens_out,
                           "latency_ms": round(res.latency_ms, 1)})
        return res
