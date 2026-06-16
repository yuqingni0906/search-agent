"""
My Third Agent — Robust Search Assistant (Week 1 result)
========================================================

This version adds no new tools and no new concepts. It just polishes the
previous my_second_agent.py from "it runs" to "it's robust and pleasant to
use." Every change is marked with "★ Week1" so you can diff it against the
old file and see exactly what changed and why.

Changelog:
  ★1  Dynamic date        -- stop hard-coding the date; use today's date automatically.
  ★2  System prompt       -- tell the model "trust the search results, not old memory".
  ★3  Cross-turn memory   -- move `messages` outside the loop so follow-up questions work.
  ★4  Tool try/except     -- if the tool (search) fails, don't crash; hand the error back to the model as the result.
  ★5  Max-rounds cap       -- prevent the model from looping forever and burning tokens.
  ★6  Model-call try/except -- if even the model is unreachable (network fully down),
                              abort this turn and roll back history instead of crashing the whole program. (added this round)

Before running (same as last time; no need to re-set keys if you already did):
    pip install anthropic tavily-python
    python my_third_agent_en.py
"""

import json
from datetime import date          # ★1 used to get "today"
from anthropic import Anthropic
from tavily import TavilyClient

client = Anthropic()               # reads the ANTHROPIC_API_KEY environment variable
tavily = TavilyClient()            # reads the TAVILY_API_KEY environment variable

MODEL = "claude-haiku-4-5-20251001"

MAX_ROUNDS = 10                    # ★5 max tool-use rounds allowed within a single question


# ============================================================
# Part 1: Define the tool -- web search (almost identical to the
#         previous version, only try/except was added)
# ============================================================

def web_search(query: str) -> str:
    """Real tool: search the web with Tavily and return a summary of the top results."""
    # ★4 This layer guards against "the tool itself failing": Tavily down,
    #    key expired, query timeout, etc. In these cases the model is actually
    #    still reachable, so we hand the error back as the "tool result" and
    #    let the model explain it gracefully instead of crashing.
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
        return f"(Search failed: {e}. Please tell the user honestly that the search did not succeed; do not make up results.)"


# The "manual" the model reads
TOOLS = [
    {
        "name": "web_search",
        "description": "Search the internet for up-to-date information. Use this tool "
                       "whenever the question involves current events, the latest data, "
                       "or facts you are unsure about. Do not answer from memory.",
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
    }
]

TOOL_FUNCTIONS = {
    "web_search": web_search,
}


# ============================================================
# ★2 System prompt -- every call tells the model: today's date + trust the search results
# ============================================================

def build_system_prompt() -> str:
    today = date.today()                       # ★1 fetch today automatically, never hard-coded
    return (
        f"Today is {today}.\n"
        "You are an assistant that can search the web. Whenever a question involves "
        "current events, the latest information, prices, or recent happenings, you must "
        "call the web_search tool and rely on the search results.\n"
        "Do not rely on the old information memorized during training -- it may be outdated.\n"
        "If the search fails or finds nothing, tell the user honestly and never fabricate."
    )


# ============================================================
# Part 2: The agent loop -- handle "one question", which may call tools over several rounds
# ============================================================

def run_one_turn(messages: list) -> str:
    """
    Handle one question from the user. The model may need to call tools over
    several rounds. Each step of the model's reply is appended IN PLACE to
    `messages` (this is the source of the ★3 memory).
    Returns: the model's final text answer.

    Note: this function does NOT catch network exceptions from
    client.messages.create -- those are handled centrally in main() (see ★6),
    because only main() knows how to roll back the history.
    """
    rounds = 0
    while True:
        rounds += 1
        if rounds > MAX_ROUNDS:                # ★5 stop past the cap so we don't loop forever and burn tokens
            return f"(Reached the max of {MAX_ROUNDS} rounds; stopping automatically to avoid an infinite loop.)"

        response = client.messages.create(     # <- when the network is down, the exception is raised here
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
                print(f"  🔧 The model decided to call a tool: {block.name}({block.input})")
                result = func(**block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })
        messages.append({"role": "user", "content": tool_results})


# ============================================================
# Part 3: Main program -- continuous conversation (★3), with model-call failures caught (★6)
# ============================================================

def main():
    messages = []                              # ★3 history lives outside the loop, kept across questions

    print("Robust search assistant started. Type a question; enter quit / exit to stop.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit", "q", ""):
            print("Goodbye!")
            break

        # ★6 Key idea: remember how long the history was BEFORE this question.
        #    If this turn fails, delete everything this turn pushed in, restoring
        #    history to a clean state.
        #    Otherwise: on failure, messages would keep a user message with no reply;
        #    next time you append another user message, you'd get two user messages
        #    in a row -- which the API forbids, causing a confusing error next time.
        snapshot = len(messages)
        messages.append({"role": "user", "content": question})

        try:
            answer = run_one_turn(messages)
            print(f"\nAssistant: {answer}\n")
        except Exception as e:
            del messages[snapshot:]            # roll back: drop all half-finished records from this turn
            print(f"\n⚠️  This request did not succeed: {e}")
            print("   It's probably a network issue (can't reach the model). Check your connection and just ask again in a moment.")
            print("   Don't worry, your earlier conversation history is unaffected.\n")


if __name__ == "__main__":
    main()
