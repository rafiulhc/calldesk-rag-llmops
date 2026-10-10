"""G2b: a better claim checker - the LLM extracts claims + evidence quotes, CODE decides.

Run:  uv run rag/judge.py      measures each judge on a labelled set of 12 drafts
"""
import re
import time
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

import agent_scratch as tools
from request_log import PRICES, cost_usd


# ---------- Judge v1 (G2): the LLM decides "supported" by itself ----------
class Verdict(BaseModel):
    supported: bool = Field(description="True only if EVERY factual claim is supported by the tool results")
    unsupported_claims: list[str] = Field(description="Each claim in the draft that the tool results do not support")


V1_INSTRUCTIONS = """You are a strict fact-checker for Calldesk answers.
Compare the DRAFT ANSWER with the TOOL RESULTS. List every factual claim in the draft (prices, plans,
features, policies, numbers) that is NOT directly supported by the tool results. Use ONLY the tool
results, not your own knowledge. Arithmetic is supported if its inputs are supported and it is correct.
Greetings, offers of help and "I don't know" are not claims. If there are no unsupported claims, supported=true."""


# ---------- Judge v2 (G2b): the LLM extracts claims and quotes; code checks them ----------
class Claim(BaseModel):
    claim: str = Field(description="One factual claim from the draft")
    kind: Literal["fact", "calculation"] = Field(description="calculation = a number the draft computed")
    evidence_quote: str = Field(description="An EXACT quote copied from the tool results that supports the claim, "
                                            "or an empty string if nothing supports it")


class Claims(BaseModel):
    claims: list[Claim]


V2_INSTRUCTIONS = """Extract every factual claim from the DRAFT ANSWER (prices, plans, features, policies, numbers).
Skip greetings, offers of help and "I don't know". For each claim:
- kind = "calculation" if the draft computed the number (totals, yearly prices, sums), otherwise "fact".
- evidence_quote = copy the exact words from the TOOL RESULTS that support it, or "" if nothing does.
Copy quotes exactly. Never invent a quote. Do not use your own knowledge."""


def normalize(text):
    return re.sub(r"\s+", " ", text.lower().replace("’", "'").replace('"', "")).strip()


def numbers_in(text):
    return {float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*\.?\d*", text)}


PLAN_NAMES = ("starter", "growth", "scale")


def check_claim(claim, evidence, calculator_results, strict=False):
    """Code, not the LLM, decides whether a claim is supported.
    v2 (strict=False): the quote must exist / a number must match a calculator result.
    v3 (strict=True):  the quote must exist AND match the claim's numbers and plan names."""
    claim_numbers = numbers_in(claim.claim)
    if claim.kind == "calculation":
        if not strict:
            return bool(claim_numbers & calculator_results)          # does it match a real calculator output?
        allowed = calculator_results | numbers_in(evidence)          # results AND their inputs are fine
        return bool(claim_numbers) and claim_numbers <= allowed
    quote = normalize(claim.evidence_quote)
    if len(quote) < 8 or quote not in normalize(evidence):           # does the quote really exist?
        return False
    if not strict:
        return True
    plans_in_claim = {p for p in PLAN_NAMES if p in normalize(claim.claim)}
    return claim_numbers <= numbers_in(quote) and plans_in_claim <= {p for p in PLAN_NAMES if p in quote}


class Judge:
    def __init__(self, version, model):
        self.version, self.model = version, model
        llm = ChatOpenAI(model=model, timeout=60, max_retries=2)
        schema = Verdict if version == "v1" else Claims
        self.llm = llm.with_structured_output(schema, include_raw=True)
        self.instructions = V1_INSTRUCTIONS if version == "v1" else V2_INSTRUCTIONS   # v3 uses the v2 prompt
        self.name = f"{version} / {model}"

    def __call__(self, draft, evidence, calculator_results=frozenset()):
        """Returns (supported, unsupported_claims, (input_tokens, output_tokens))."""
        out = self.llm.invoke([SystemMessage(self.instructions),
                               HumanMessage(f"TOOL RESULTS:\n{evidence or '(no tools were called)'}\n\nDRAFT ANSWER:\n{draft}")])
        usage = out["raw"].usage_metadata
        tokens = (usage["input_tokens"], usage["output_tokens"])
        if self.version == "v1":
            return out["parsed"].supported, out["parsed"].unsupported_claims, tokens
        strict = self.version == "v3"
        bad = [c.claim for c in out["parsed"].claims if not check_claim(c, evidence, set(calculator_results), strict)]
        return not bad, bad, tokens


# ---------- A labelled set: drafts a human has marked supported (True) or not (False) ----------
def labelled_cases():
    growth = f"[tool: get_plan] {tools.get_plan('growth')}"
    pricing = f"[tool: search_docs] {tools.search_docs('Which plan includes voicemail transcription?')}"
    support = f"[tool: search_docs] {tools.search_docs('Does Calldesk make outbound calls? Which countries?')}"
    support_levels = f"[tool: search_docs] {tools.search_docs('Which plan has 24/7 support?')}"
    annual = f"[tool: search_docs] {tools.search_docs('annual billing discount')}"
    calc_bill = f"[tool: calculate] {tools.calculate('899 + 3 * 100')}"
    calc_year = f"[tool: calculate] {tools.calculate('899 * 12 * 0.8')}"
    return [
        ("The Growth plan costs $899 per month and includes 1,200 minutes.", growth, True),
        ("The Growth plan costs $799 per month.", growth, False),
        ("The Growth plan includes voicemail transcription [pricing.txt #1].", pricing, True),
        ("The Starter plan includes voicemail transcription.", pricing, False),
        ("Your total monthly bill would be $1,199 (899 + 3 x 100).", f"{growth}\n\n{calc_bill}", True),
        ("Your total monthly bill would be $1,299.", f"{growth}\n\n{calc_bill}", False),
        ("Calldesk does not make outbound calls.", support, True),
        ("Calldesk works for businesses in Canada.", support, False),
        ("Hello! How can I help you today?", "", True),
        ("Paying annually gives 20% off, so Growth costs $8,630.40 per year.", growth, False),
        ("Paying annually gives 20% off, so Growth costs $8,630.40 per year.", f"{growth}\n\n{annual}\n\n{calc_year}", True),
        ("The Scale plan includes dedicated 24/7 support.", support_levels, True),
    ]


def calculator_results_in(evidence):
    return {float(x) for x in re.findall(r"\[tool: calculate\] ([\d.]+)", evidence)}


if __name__ == "__main__":
    cases = labelled_cases()
    judges = [Judge("v1", "gpt-4o-mini"), Judge("v2", "gpt-4o-mini"), Judge("v2", "gpt-5.4-mini"),
              Judge("v3", "gpt-4o-mini"), Judge("v3", "gpt-5.4-mini")]
    PRICES.setdefault("gpt-5.4-mini", {"input": 0.75, "output": 4.50})
    RUNS = 2

    print(f"{len(cases)} labelled drafts ({sum(c[2] for c in cases)} supported, {sum(not c[2] for c in cases)} unsupported), "
          f"{RUNS} runs per judge\n")
    for judge in judges:
        correct = false_reject = false_accept = 0
        t_in = t_out = 0
        start = time.perf_counter()
        mistakes = set()
        for _ in range(RUNS):
            for draft, evidence, expected in cases:
                supported, bad, (i, o) = judge(draft, evidence, calculator_results_in(evidence))
                t_in, t_out = t_in + i, t_out + o
                if supported == expected:
                    correct += 1
                elif expected:
                    false_reject += 1
                    mistakes.add(f"false REJECT: {draft[:60]!r}  (judge said: {bad})")
                else:
                    false_accept += 1
                    mistakes.add(f"false ACCEPT: {draft[:60]!r}")
        n = RUNS * len(cases)
        cost = cost_usd(judge.model, t_in, t_out)
        print(f"{judge.name:<20} accuracy {100 * correct / n:>3.0f}%   false rejects {false_reject}   "
              f"false accepts {false_accept}   {(time.perf_counter() - start) / n:.1f}s/check   ${cost / n:.6f}/check")
        for m in sorted(mistakes):
            print(f"     {m}")
