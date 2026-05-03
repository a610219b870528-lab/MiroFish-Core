import asyncio
import time
import os
from google import genai
from google.genai import types

# 🚨 最大併發數限制，防止打爆 API 限流
MAX_CONCURRENT_REQUESTS = 10
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

client = genai.Client()

MOCK_SOCIETY = [
    {"id": "A001", "role": "八卦版鄉民_小明", "tier": "flash"},
    {"id": "A002", "role": "吃瓜群眾_阿華", "tier": "flash"},
    {"id": "A003", "role": "憤怒的消費者_老李", "tier": "flash"},
    {"id": "K001", "role": "財經KOL_輝哥", "tier": "pro"},
    {"id": "K002", "role": "品牌官方_公關部", "tier": "pro"},
]


async def call_gemini_api_real(
    agent_id: str, role: str, tier: str, event_context: str
) -> dict:
    async with semaphore:
        start_time = time.time()

        # 👑 2026 動態算力路由 (Gemini 2.5 矩陣)
        primary_model = "gemini-2.5-pro" if tier == "pro" else "gemini-2.5-flash-lite"
        fallback_model = "gemini-2.5-flash"

        system_instruction = f"你是 MiroFish 宇宙中的角色：{role}。請完全進入角色，用語氣強烈、符合身份的方式，用一句話回應目前發生的事件。不要說廢話。"

        try:
            response = await client.aio.models.generate_content(
                model=primary_model,
                contents=event_context,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.8,
                    max_output_tokens=150,
                ),
            )
            action = response.text.strip()
            actual_model = primary_model

        except Exception as primary_e:
            primary_error_msg = str(primary_e)
            print(
                f"⚠️ [路由切換] {primary_model} 異常，啟動備援: {fallback_model} (Agent: {agent_id})"
            )
            try:
                response = await client.aio.models.generate_content(
                    model=fallback_model,
                    contents=event_context,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.8,
                        max_output_tokens=150,
                    ),
                )
                action = response.text.strip()
                actual_model = f"{fallback_model} (Fallback)"
            except Exception as fallback_e:
                action = (
                    f"🚨 [崩潰] 主路由: {primary_error_msg} | 備援: {str(fallback_e)}"
                )
                actual_model = "error"

        latency = round(time.time() - start_time, 2)

        return {
            "agent_id": agent_id,
            "role": role,
            "model_used": actual_model,
            "action": action,
            "latency_sec": latency,
        }


# 🚨 CTO 更新：這裡改為接收外部傳入的 event_context
async def run_simulation(event_context: str) -> dict:
    print("🚨 [CTO 廣播] 接收到外部請求，沙盤推演啟動...")
    start_time = time.time()

    print(f"📊 動態推演事件：{event_context}")

    tasks = [
        call_gemini_api_real(agent["id"], agent["role"], agent["tier"], event_context)
        for agent in MOCK_SOCIETY
    ]

    results = await asyncio.gather(*tasks)

    total_time = round(time.time() - start_time, 2)
    print(f"✅ 推演完成！耗時 {total_time} 秒。")

    return {
        "status": "success",
        "event_simulated": event_context,
        "total_agents": len(results),
        "execution_time_sec": total_time,
        "details": results,
    }
