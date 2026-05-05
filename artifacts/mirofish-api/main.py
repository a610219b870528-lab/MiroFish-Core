import os
import json
import asyncio
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Literal
import google.generativeai as genai

# ---------------------------------------------------------------------------
# 1. 系統初始化與資安防護 (System Init & Security)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("🚨 致命錯誤：未檢測到 GEMINI_API_KEY。")
else:
    genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(
    title="MiroFish Universal Engine (內建 AI 選角導演版)",
    description="""
    **兩階段流水線：**
    1. 使用 `/generate_cast` 讓 AI 自動生成百人角色清單。
    2. 將清單送入 `/simulate` 進行 Map-Reduce 巨型沙盤推演。
    """,
    version="4.1.0",
)


# ---------------------------------------------------------------------------
# 2. 雙語化資料模型 (含選角導演專用 Schema)
# ---------------------------------------------------------------------------
class AgentRole(BaseModel):
    role_name: str = Field(..., description="角色名稱 (Role Name)")
    stance: str = Field(
        ..., description="角色的基本立場或背景設定 (Stance or Background)"
    )
    tier: Literal["flash", "pro"] = Field(
        "flash", description="算力分級：flash (配角) / pro (主角)"
    )


# 👑 新增：選角導演的請求模型
class GenerateCastRequest(BaseModel):
    event_timeline: List[str] = Field(
        ..., description="事件時間軸陣列 (Array of chronological events)"
    )
    pro_count: int = Field(2, description="需要 AI 生成幾個『高階決策主角 (Pro)』？")
    flash_count: int = Field(
        10, description="需要 AI 生成幾個『基層配角/群眾 (Flash)』？"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "event_timeline": ["印刷廠四色機突發燒毀，無法接急單。"],
                    "pro_count": 2,
                    "flash_count": 5,
                }
            ]
        }
    }


class SimulateRequest(BaseModel):
    event_timeline: List[str] = Field(..., description="事件時間軸陣列")
    agents: List[AgentRole] = Field(..., description="動態角色列表")


class SimulateResponse(BaseModel):
    status: str
    map_reduce_triggered: bool
    executive_summary: str
    agent_reports: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# 3. 核心引擎 (選角 + 推演 + Map-Reduce)
# ---------------------------------------------------------------------------
MAX_CONCURRENT_REQUESTS = 15
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
REDUCE_CHUNK_SIZE = 20


# ================= 新增：AI 選角導演邏輯 =================
async def generate_roster_via_ai(
    timeline: List[str], pro_count: int, flash_count: int
) -> List[Dict]:
    """利用 Gemini 直接生成符合 AgentRole 結構的 JSON 陣列"""
    formatted_timeline = "\n".join([f"- {event}" for event in timeline])

    # 強制要求 Gemini 回傳 JSON 格式
    generation_config = genai.GenerationConfig(response_mime_type="application/json")

    system_prompt = """你是一個專業的商業沙盤『選角導演』。
    請根據傳入的事件，設計出符合衝突與推演邏輯的利害關係人。
    你必須嚴格回傳一個 JSON 陣列 (Array)，裡面包含物件。
    每個物件必須有三個字串 key："role_name" (角色名稱), "stance" (具體立場與痛點), "tier" ("pro" 或 "flash")。
    """

    user_prompt = f"事件時間軸：\n{formatted_timeline}\n\n請幫我生成 {pro_count} 個高階決策主角 (tier='pro')，以及 {flash_count} 個會受影響的基層配角或群眾 (tier='flash')。確保他們的立場互相衝突或多樣化。"

    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=system_prompt,
            generation_config=generation_config,
        )
        response = await model.generate_content_async(user_prompt)
        # 解析 Gemini 吐出的 JSON 字串轉為 Python List
        agents_list = json.loads(response.text)
        return agents_list
    except Exception as e:
        logger.error(f"🚨 選角生成失敗: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI 選角失敗: {str(e)}")


# ================= 既有：推演與壓縮邏輯 =================
async def call_gemini_agent(agent: AgentRole, timeline: List[str]) -> dict:
    model_name = "gemini-2.5-pro" if agent.tier == "pro" else "gemini-2.5-flash"
    formatted_timeline = "\n".join([f"- {event}" for event in timeline])

    system_prompt = f"你現在是【{agent.role_name}】。你的背景與立場是：【{agent.stance}】。拒絕廢話，直指核心。"
    user_prompt = (
        f"事件時間軸：\n{formatted_timeline}\n\n請基於你的立場給出具體反應（條列式）。"
    )

    async with semaphore:
        try:
            model = genai.GenerativeModel(
                model_name=model_name, system_instruction=system_prompt
            )
            response = await model.generate_content_async(user_prompt)
            return {
                "role": agent.role_name,
                "stance": agent.stance,
                "tier": agent.tier,
                "action": response.text,
            }
        except Exception as e:
            return {"role": agent.role_name, "tier": agent.tier, "error": str(e)}


async def reduce_flash_reports(
    chunk_reports: List[Dict], timeline: List[str], chunk_id: int
) -> str:
    model_name = "gemini-2.5-flash"
    formatted_timeline = "\n".join([f"- {event}" for event in timeline])
    raw_text = "\n".join(
        [f"【{r['role']}】: {r.get('action', '失敗')}" for r in chunk_reports]
    )
    prompt = f"事件時間軸：\n{formatted_timeline}\n\n以下是第 {chunk_id} 批基層群眾的反應：\n{raw_text}\n\n請濃縮成 300 字以內的情緒風向與重點。"
    async with semaphore:
        try:
            model = genai.GenerativeModel(model_name=model_name)
            response = await model.generate_content_async(prompt)
            return f"【秘書風向匯報 - 批次 {chunk_id}】:\n{response.text}"
        except Exception as e:
            return f"【秘書匯報失敗 - 批次 {chunk_id}】"


async def call_gemini_pro_summary(timeline: List[str], final_context: str) -> str:
    model_name = "gemini-2.5-pro"
    formatted_timeline = "\n".join([f"- {event}" for event in timeline])
    system_prompt = "你是本案的最高決策總監。請根據『事件發展』、『主角意見』與『群眾風向』，統整出解決危機的【最終戰略 SOP 表格】。使用繁體中文。"
    user_prompt = f"事態時間軸：\n{formatted_timeline}\n\n情報匯總：\n{final_context}\n\n請輸出決策報表。"
    try:
        model = genai.GenerativeModel(
            model_name=model_name, system_instruction=system_prompt
        )
        response = await model.generate_content_async(user_prompt)
        return response.text
    except Exception as e:
        return "統整失敗 (Summary Failed)。"


# ---------------------------------------------------------------------------
# 4. 兩階段 API 端點路由 (Two-Stage Endpoints)
# ---------------------------------------------------------------------------
@app.get("/", summary="系統狀態")
async def health_check():
    return {"status": "online", "version": "4.1.0 (AI Casting Edition)"}


# 👑 階段一：選角導演
@app.post(
    "/generate_cast",
    summary="Phase 1: 讓 AI 模擬並生成角色清單 (Auto-Cast)",
    response_model=List[AgentRole],
)
async def auto_generate_cast(request: GenerateCastRequest):
    """輸入事件與需求人數，AI 將自動生成對應的利益關係人清單，供操作員參考或微調。"""
    agents = await generate_roster_via_ai(
        request.event_timeline, request.pro_count, request.flash_count
    )
    return agents


# 👑 階段二：正式推演
@app.post(
    "/simulate",
    response_model=SimulateResponse,
    summary="Phase 2: 執行百人沙盤推演 (Simulation)",
)
async def run_universal_simulation(request: SimulateRequest):
    """將生成好的角色清單送入，進行 Map-Reduce 巨型推演。"""
    tasks = [
        call_gemini_agent(agent, request.event_timeline) for agent in request.agents
    ]
    all_reports = await asyncio.gather(*tasks)

    pro_reports = [r for r in all_reports if r.get("tier") == "pro"]
    flash_reports = [r for r in all_reports if r.get("tier") == "flash"]

    final_context_builder = "=== 核心主角 (Pro) 意見 ===\n"
    for r in pro_reports:
        final_context_builder += f"【{r['role']}】:\n{r.get('action', '錯誤')}\n---\n"

    final_context_builder += "=== 基層群眾 (Flash) 風向 ===\n"
    map_reduce_triggered = False
    if len(flash_reports) <= REDUCE_CHUNK_SIZE:
        for r in flash_reports:
            final_context_builder += (
                f"【{r['role']}】:\n{r.get('action', '錯誤')}\n---\n"
            )
    else:
        map_reduce_triggered = True
        chunks = [
            flash_reports[i : i + REDUCE_CHUNK_SIZE]
            for i in range(0, len(flash_reports), REDUCE_CHUNK_SIZE)
        ]
        reduce_tasks = [
            reduce_flash_reports(chunk, request.event_timeline, i + 1)
            for i, chunk in enumerate(chunks)
        ]
        summarized_chunks = await asyncio.gather(*reduce_tasks)
        final_context_builder += "\n\n".join(summarized_chunks)

    executive_summary = await call_gemini_pro_summary(
        request.event_timeline, final_context_builder
    )

    return SimulateResponse(
        status="success",
        map_reduce_triggered=map_reduce_triggered,
        executive_summary=executive_summary,
        agent_reports=all_reports,
    )
