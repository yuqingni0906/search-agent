"""
My Fourth Agent — Multi-Tool Research Assistant (search + fetch + calculate + docs)
===================================================================================

Building on the "robust search assistant," this version adds a SECOND tool and beyond,
teaching the model to choose among multiple tools: first web_search to find links →
then fetch_page to read the full article. That is the dividing line of a "real agent":
the decision power is upgraded from "whether to call a tool" to "which tool, in what order."

Changelog (★1–★6 are robustness from Week 1; ★7 is Week 3's tool; ★8–★9 are Week 4):
  ★1  Dynamic date         — no hard-coded date; always uses today, automatically
  ★2  system prompt        — tells the model "trust the search results, not stale memory"
  ★3  Cross-question memory — messages lifted outside the loop, so follow-ups work
  ★4  Tool try/except       — a failing tool doesn't crash; the error is returned as a result
  ★5  Max-rounds cap        — stops the model from looping forever and burning tokens
  ★6  Model-call try/except — on connection failure, abort this turn and roll back history
  ★7  Second tool fetch_page — scrapes page text, enabling "search first, then read deeply"
  ★8  Third tool calculate   — exact arithmetic; insurance against the model fabricating numbers
  ★9  Tools 4 & 5: doc Q&A   — list_files (discover) + read_file (read), "ask my documents"

Setup before running (requests / beautifulsoup4 added in Week 3; pypdf optional for PDFs):
    pip install anthropic tavily-python requests beautifulsoup4 pypdf
    python my_4th_agent_multiTool.py
"""

import json
from datetime import date          # ★1 used to get "today"
import requests                     # ★7 fetch_page uses this to scrape pages
from bs4 import BeautifulSoup       # ★7 fetch_page uses this to parse the body text
from anthropic import Anthropic
from tavily import TavilyClient
import ast
import operator
import os

client = Anthropic()               # reads env var ANTHROPIC_API_KEY
tavily = TavilyClient()            # reads env var TAVILY_API_KEY

MODEL = "claude-haiku-4-5-20251001"

MAX_ROUNDS = 10                    # ★5 in a single question, the max tool-calling rounds allowed


# ============================================================
# Part 1: Define tools — web search (nearly identical to the last version, just added try/except)
# ============================================================

def web_search(query: str) -> str:
    """Real tool: search the web via Tavily, return summaries of the top results."""
    # ★4 This layer guards against "the tool itself failing": Tavily is down, the key is
    #    invalid, the query times out... The model is actually still reachable here, so we
    #    hand the error back as a "tool result" and let the model explain it gracefully to
    #    the user, rather than crashing.
    try:
        response = tavily.search(query=query, max_results=3)
        results = []
        for item in response["results"]:
            results.append(
                f"Title: {item['title']}\n"
                f"Content: {item['content']}\n"
                f"Source: {item['url']}"
            )
        return "\n\n---\n\n".join(results)
    except Exception as e:
        return f"(Search failed: {e}. Please tell the user honestly that the search did not succeed; do not fabricate results.)"


def fetch_page(url: str) -> str:
    """Scrape the body text of one web page; on failure, return the error as a 'result' instead of raising."""
    try:
        # Many sites reject the default python-requests UA, so we spoof a browser User-Agent.
        headers = {"User-Agent": "Mozilla/5.0 (learning-agent)"}
        # timeout is mandatory; without it, an unresponsive site would hang forever.
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()   # treat 4xx / 5xx as errors directly

        soup = BeautifulSoup(resp.text, "html.parser")
        # strip out script, style, nav, footer — the noise
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)

        return text[:5000]   # truncate! Dumping a whole page wastes tokens and can overflow length limits
    except Exception as e:
        return f"Fetch failed: {e}"   # same habit: an error is also a result, handed back to the model to judge



# ============================================================
# ★8 Third tool: calculate — exact arithmetic, insurance against the model "making up numbers"
#    Uses ast parsing + a whitelist, allowing only +-*/ etc., NEVER executing arbitrary code (unlike eval)
# ============================================================

_ALLOWED = {
    ast.Add: operator.add,   ast.Sub: operator.sub,
    ast.Mult: operator.mul,  ast.Div: operator.truediv,
    ast.Pow: operator.pow,   ast.Mod: operator.mod,
    ast.USub: operator.neg,
}

def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        op = _ALLOWED.get(type(node.op))
        if op is None: raise ValueError("Unsupported operator")
        return op(_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _ALLOWED.get(type(node.op))
        if op is None: raise ValueError("Unsupported operator")
        return op(_eval(node.operand))
    raise ValueError("Not a valid arithmetic expression")

def calculate(expression: str) -> str:
    """Safely evaluate an arithmetic expression and return the result as a string."""
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_eval(tree.body))
    except Exception as e:
        # ★4 same fault-layering: if it can't be computed, return the error as a result, don't crash
        return f"Calculation error: {type(e).__name__} — {e}"


# ============================================================
# ★9 Tools 4 & 5: document Q&A — list_files (discover) + read_file (read)
#    Same shape as Week 3's web_search→fetch_page, just moved from "the web" to "local disk"
# ============================================================

DOCS_DIR = os.path.realpath("./my_docs")   # only files inside this folder may be read

def list_files() -> str:
    """List which documents are available to read inside my_docs."""
    try:
        if not os.path.isdir(DOCS_DIR):
            return "(No my_docs folder yet. Please create one next to this program and put documents in it.)"
        names = [f for f in os.listdir(DOCS_DIR)
                 if os.path.isfile(os.path.join(DOCS_DIR, f))]
        return ("Readable documents:\n" + "\n".join(names)) if names else "(my_docs is empty.)"
    except Exception as e:
        return f"Listing failed: {type(e).__name__} — {e}"   # ★4 same: an error is returned as a result

def read_file(filename: str) -> str:
    """Read the body text of a document inside my_docs (.pdf and plain text supported); return errors as results."""
    try:
        # ★ Safety guardrail: lock the path inside my_docs, blocking "../../etc/passwd" path-traversal attacks
        target = os.path.realpath(os.path.join(DOCS_DIR, filename))
        if not target.startswith(DOCS_DIR + os.sep):
            return "(Refused: only files inside the my_docs folder can be read.)"
        if not os.path.isfile(target):
            return f"(Cannot find {filename}. Try list_files first to see what's available.)"

        if target.lower().endswith(".pdf"):
            from pypdf import PdfReader      # lazy import: no need to install this library if only reading plain text
            reader = PdfReader(target)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        else:
            with open(target, encoding="utf-8", errors="ignore") as f:
                text = f.read()

        return text[:5000] if text.strip() else "(File was read, but no text could be extracted.)"
    except Exception as e:
        return f"Read failed: {type(e).__name__} — {e}"


# The "instruction manual" the model sees
TOOLS = [
    {
        "name": "web_search",
        "description": "Search the internet for up-to-date information. Use this tool whenever the "
                       "question involves current events, the latest data, or facts you are unsure "
                       "about; do not answer from memory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "search keywords",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_page",
        "description": "Scrape the body text of a given web page URL. Use this once you have a link "
                       "from search and need to read the details of a specific full article.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "the full URL of the page to scrape"}
            },
            "required": ["url"]
        }
    },
    {
        "name": "calculate",
        "description": (
            "Compute a math expression and return the exact result. ANY arithmetic — addition, "
            "subtraction, multiplication, division, powers, parenthesized compound expressions — "
            "MUST go through this tool; even if it looks trivial, do not do it in your head. "
            "Pass a standard Python arithmetic string, e.g. '48127 * 9043' or '(123 + 456) * 7'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "the expression to compute, using + - * / ** % and parentheses",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "list_files",
        "description": "List which documents are available to read in the my_docs folder. When the user "
                       "asks about 'my documents/files' but you don't yet know what files exist, use "
                       "this first to see what's there.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "read_file",
        "description": "Read the body text of a document in my_docs (.pdf and plain text supported). "
                       "Use list_files first to learn what files exist, then use this to read the one "
                       "you need.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "the filename to read, e.g. report.pdf"}
            },
            "required": ["filename"],
        },
    }

]

TOOL_FUNCTIONS = {
    "web_search": web_search,
    "fetch_page": fetch_page,
    "calculate": calculate,
    "list_files": list_files,
    "read_file": read_file,
}


# ============================================================
# ★2 system prompt — every turn, tell the model: today's date + trust the search results
# ============================================================

def build_system_prompt() -> str:
    today = date.today()                       # ★1 fetch today automatically, no hard-coding
    return (
        f"Today is {today}.\n"
        "You are an assistant that can search the web. When a question involves current events, "
        "the latest information, prices, or recent happenings, you MUST call the web_search tool "
        "to look it up, and treat the search results as authoritative.\n"
        "Do not rely on stale information memorized during training — it may be out of date.\n"
        "If a search fails or finds nothing, tell the user honestly; never fabricate."
    )


# ============================================================
# Part 2: the agent loop — handles "one question," which may span several tool rounds
# ============================================================

def run_one_turn(messages: list) -> str:
    """
    Handle one user question. In between, the model may call tools across several rounds.
    Every model reply is appended IN PLACE to messages (this is the source of ★3 memory).
    Returns: the model's final text answer.

    Note: this function does NOT catch network exceptions from client.messages.create —
    that is handled centrally by main() (see ★6), because only main() knows how to roll
    back the history.
    """
    rounds = 0
    while True:
        rounds += 1
        if rounds > MAX_ROUNDS:                # ★5 stop past the cap, don't burn tokens forever
            return f"(Reached the max of {MAX_ROUNDS} rounds; stopping automatically to avoid an infinite loop.)"

        response = client.messages.create(     # ← on a dropped connection, this is where it raises
            model=MODEL,
            max_tokens=1024,
            system=build_system_prompt(),      # ★2
            tools=TOOLS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            final_text = ""
            for block in response.content:
                if block.type == "text":
                    final_text += block.text
            return final_text.strip()

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                func = TOOL_FUNCTIONS[block.name]
                print(f"  🔧 Model decided to call tool: {block.name}({block.input})")
                result = func(**block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })
        messages.append({"role": "user", "content": tool_results})


# ============================================================
# Part 3: main program — continuous conversation (★3), with model-call failures caught (★6)
# ============================================================

def main():
    messages = []                              # ★3 history lives outside the loop, preserved across questions

    print("Robust search assistant started. Just type your question; enter quit / exit to stop.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit", "q", ""):
            print("Goodbye!")
            break

        # ★6 Key: remember how long the history was BEFORE this question.
        #    If this turn fails, delete everything this turn pushed in, restoring a clean state.
        #    Otherwise: on failure, messages would keep a user message that nothing responded to,
        #    and the next append of another user message would put two user messages back-to-back —
        #    which the API forbids, so the next call would always error.
        snapshot = len(messages)
        messages.append({"role": "user", "content": question})

        try:
            answer = run_one_turn(messages)
            print(f"\nAssistant: {answer}\n")
        except Exception as e:
            del messages[snapshot:]            # roll back: delete all the half-finished records from this turn
            # Print the exception TYPE too — a lesson from a past pitfall:
            #   just saying "probably a network issue" fools you. The type tells them apart at a glance:
            #   · Network-class (e.g. APIConnectionError / Timeout) → truly unreachable, just retry later
            #   · Code-class (e.g. KeyError / TypeError)            → it's a bug, go fix the code
            print(f"\n⚠️  This request did not succeed: {type(e).__name__}: {e}")
            print("   If it's a network-class error, check your connection and retry later;")
            print("   if it's a KeyError / TypeError, it's likely a code bug that needs fixing.")
            print("   Don't worry — the prior conversation history was rolled back automatically, unaffected.\n")


if __name__ == "__main__":
    main()
