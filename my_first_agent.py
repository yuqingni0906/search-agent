"""
我的第一个 Agent —— 最小可运行示例
====================================

这个文件演示 agent 的核心:一个 "模型决策 → 代码执行 → 结果回传" 的循环。

运行前准备:
    1. pip install anthropic
    2. 拿一个 API key: https://console.anthropic.com/  (新账号通常有免费额度)
    3. 设置环境变量:
         Mac/Linux:  export ANTHROPIC_API_KEY="你的key"
         Windows:    set ANTHROPIC_API_KEY=你的key
    4. python my_first_agent.py

第一版用的是「假工具」(get_weather 返回写死的数据),目的是让你
亲眼看到循环转起来。看懂之后,把 get_weather 换成真的搜索/抓取即可。
"""

import json
from anthropic import Anthropic

client = Anthropic()  # 自动读取环境变量 ANTHROPIC_API_KEY

MODEL = "claude-haiku-4-5-20251001"  # 便宜够用;想效果更好可换 claude-sonnet-4-6


# ============================================================
# 第 1 部分:定义「工具」
# ------------------------------------------------------------
# 工具 = 一个普通的 Python 函数 + 一份给模型看的「说明书」。
# 模型本身不会执行代码,它只会读说明书后告诉你「我想调用这个工具,参数是这些」,
# 真正执行的是你下面这个函数。
# ============================================================

def get_weather(city: str) -> str:
    """假工具:先返回写死的数据,用来跑通流程。
    跑通后你可以把这里换成真正的搜索/抓取(见文件底部说明)。"""
    fake_data = {
        "北京": "晴,18°C",
        "上海": "多云,22°C",
        "广州": "小雨,27°C",
    }
    return fake_data.get(city, f"暂无 {city} 的天气数据")


# 给模型看的「说明书」。description 和参数描述写得越清楚,模型调用得越准。
TOOLS = [
    {
        "name": "get_weather",
        "description": "查询某个城市当前的天气情况。当用户询问天气时使用。",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称,例如 北京、上海",
                }
            },
            "required": ["city"],
        },
    }
]

# 把工具名映射到真正的函数,方便循环里按名字调用
TOOL_FUNCTIONS = {
    "get_weather": get_weather,
}


# ============================================================
# 第 2 部分:Agent 循环
# ------------------------------------------------------------
# 这是整个 agent 的心脏。流程:
#   1. 把对话发给模型
#   2. 模型要么直接回答(stop_reason != "tool_use"),要么要求调工具
#   3. 如果要调工具:我们执行它,把结果塞回对话,再回到第 1 步
#   4. 直到模型给出最终答案
# ============================================================

def run_agent(user_message: str) -> str:
    # messages 保存完整对话历史。模型没有记忆,每次都要把全部历史发过去。
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )

        # 把模型这一轮的回复(可能包含工具调用请求)加进历史
        messages.append({"role": "assistant", "content": response.content})

        # 如果模型不再要求调工具,说明它给出了最终答案,结束循环
        if response.stop_reason != "tool_use":
            # 取出文本部分返回
            final = "".join(
                block.text for block in response.content if block.type == "text"
            )
            return final

        # 否则:模型要求调用一个或多个工具,我们逐个执行
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                func = TOOL_FUNCTIONS[block.name]
                print(f"  🔧 模型决定调用工具: {block.name}({json.dumps(block.input, ensure_ascii=False)})")
                result = func(**block.input)   # 真正执行你的 Python 函数
                print(f"  📦 工具返回: {result}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

        # 把工具执行结果作为「user」消息塞回去,让模型据此继续
        messages.append({"role": "user", "content": tool_results})


# ============================================================
# 第 3 部分:试运行
# ============================================================

if __name__ == "__main__":
    question = "北京和广州现在天气怎么样?哪个更适合出门?"
    print(f"❓ 用户提问: {question}\n")
    answer = run_agent(question)
    print(f"\n✅ 最终回答:\n{answer}")
