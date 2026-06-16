"""
我的第三个 Agent —— 健壮版搜索助手 
==================================================

这一版没有新工具、没有新概念,只是把上一版 my_second_agent.py
从「能跑」打磨到「健壮、好用」。改动全部用「★ 第1周」标出。
你对着旧文件看,就知道每一处动了什么、为什么动。

改动清单:
  ★1  动态日期       —— 不再写死日期,每天自动用今天
  ★2  system prompt  —— 告诉模型「以搜索结果为准,别信旧记忆」
  ★3  跨提问记忆     —— messages 提到循环外,可以连续追问
  ★4  工具 try/except —— 工具(搜索)失败不崩溃,把错误当结果交回给模型
  ★5  最大轮次上限   —— 防止模型陷入死循环、无限烧 token
  ★6  模型调用 try/except —— 连模型都连不上(网络全断)时,中止这轮并回滚历史,
                            而不是整个程序崩掉。← 本次新补

运行前准备(和上次一样,key 之前设过就不用再设):
    pip install anthropic tavily-python
    python my_third_agent.py
"""

import json
from datetime import date          # ★1 用来取「今天」
from anthropic import Anthropic
from tavily import TavilyClient

client = Anthropic()               # 读环境变量 ANTHROPIC_API_KEY
tavily = TavilyClient()            # 读环境变量 TAVILY_API_KEY

MODEL = "claude-haiku-4-5-20251001"

MAX_ROUNDS = 10                    # ★5 一次提问里,最多让模型调几轮工具


# ============================================================
# 第 1 部分:定义工具 —— 网页搜索(和上一版几乎一样,只加了 try/except)
# ============================================================

def web_search(query: str) -> str:
    """真工具:用 Tavily 搜索网页,返回前几条结果的摘要。"""
    # ★4 这层守的是「工具自己出问题」:Tavily 挂了、key 失效、查询超时……
    #    模型这时其实是连得上的,所以把错误当成「工具结果」交回去,
    #    让模型优雅地跟用户解释,而不是崩溃。
    try:
        response = tavily.search(query=query, max_results=3)
        results = []
        for item in response["results"]:
            results.append(
                f"标题: {item['title']}\n"
                f"内容: {item['content']}\n"
                f"来源: {item['url']}"
            )
        return "\n\n---\n\n".join(results)
    except Exception as e:
        return f"(搜索失败:{e}。请如实告诉用户搜索没成功,不要编造结果。)"


# 给模型看的「说明书」
TOOLS = [
    {
        "name": "web_search",
        "description": "搜索互联网获取最新信息。当问题涉及时事、最新数据、"
                       "或你不确定的事实时,用这个工具去查,不要凭记忆回答。",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词",
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
# ★2 system prompt —— 每次都告诉模型:今天几号 + 以搜索结果为准
# ============================================================

def build_system_prompt() -> str:
    today = date.today()                       # ★1 自动取今天,不写死
    return (
        f"今天是 {today}。\n"
        "你是一个能联网搜索的助手。当问题涉及时事、最新信息、价格、"
        "近期发生的事时,必须调用 web_search 工具去查,并以搜索结果为准。\n"
        "不要依赖你训练时记下的旧信息——那些可能已经过时。\n"
        "如果搜索失败或查不到,如实告诉用户,绝不编造。"
    )


# ============================================================
# 第 2 部分:agent 循环 —— 处理「一次提问」,中间可能调好几轮工具
# ============================================================

def run_one_turn(messages: list) -> str:
    """
    处理用户的一次提问。中间模型可能要调好几轮工具。
    把模型每一步回复都【原地追加】进 messages(这就是 ★3 记忆的来源)。
    返回:模型最终的文字回答。

    注意:这里【不】捕获 client.messages.create 的网络异常 ——
    那个交给 main() 统一处理(见 ★6),因为只有 main() 知道怎么回滚历史。
    """
    rounds = 0
    while True:
        rounds += 1
        if rounds > MAX_ROUNDS:                # ★5 超过上限就停,别无限烧 token
            return f"(已达到最大轮次 {MAX_ROUNDS},自动停止,避免死循环。)"

        response = client.messages.create(     # ← 网络断时,会在这里抛异常
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
                print(f"  🔧 模型决定调用工具: {block.name}({block.input})")
                result = func(**block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })
        messages.append({"role": "user", "content": tool_results})


# ============================================================
# 第 3 部分:主程序 —— 连续对话(★3),并兜住模型调用失败(★6)
# ============================================================

def main():
    messages = []                              # ★3 历史放循环外,跨提问保留

    print("健壮版搜索助手已启动。直接打字提问,输入 quit / 退出 结束。\n")
    while True:
        question = input("你:").strip()
        if question.lower() in ("quit", "exit", "退出", "q", ""):
            print("再见!")
            break

        # ★6 关键:记住「这次提问之前」历史有多长。
        #    万一这轮失败,就把这轮往里塞的东西全删掉,把历史还原到干净状态。
        #    否则:失败时 messages 里会留下一条没人回应的 user 消息,
        #    下次再 append 一条 user,就出现两条 user 连着 —— API 不允许,下次必报错。
        snapshot = len(messages)
        messages.append({"role": "user", "content": question})

        try:
            answer = run_one_turn(messages)
            print(f"\n助手:{answer}\n")
        except Exception as e:
            del messages[snapshot:]            # 回滚:删掉这轮所有半截的记录
            print(f"\n⚠️  这次请求没成功:{e}")
            print("   多半是网络问题(连不上模型)。检查下网络,等会儿再问一次就行。")
            print("   放心,之前的对话历史没受影响。\n")


if __name__ == "__main__":
    main()
