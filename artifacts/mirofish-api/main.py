import os
import json
import asyncio
import logging
import time
import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Literal
import redis.asyncio as redis

# 👑 CTO 架構升級：引入最新版 Google GenAI SDK
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# 1. 系統初始化與資安防護 (System Init & Security)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
REDIS_URL = os.getenv("REDIS_URL")

if not GEMINI_API_KEY:
    raise ValueError("🚨 致命錯誤：未檢測到 GEMINI_API_KEY。")
if not REDIS_URL:
    raise ValueError("🚨 致命錯誤：未檢測到 REDIS_URL，Redis 緩存無法啟動。")

# 初始化新版 Gemini 客戶端與 Redis 連線池
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

app = FastAPI(
    title="MiroFish Universal Engine (v4.3.0 商用容錯版)",
    description="全面升級 google-genai SDK，並實裝 Redis 雙保險快取機制與查詢端點。",
    version="4.3.0",
)

from fastapi.middleware.cors import CORSMiddleware

# 👑 CTO 資安設定：打通跨域防線 (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允許所有前端網址呼叫 (上線後可鎖定專屬網域)
    allow_credentials=True,
    allow_methods=["*"],  # 允許 GET, POST 等所有方法
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# 2. 雙語化資料模型 (Bilingual Schemas)
# ---------------------------------------------------------------------------
class AgentRole(BaseModel):
    role_name: str = Field(..., description="角色名稱 (Role Name)")
    stance: str = Field(..., description="角色的基本立場或背景設定 (Stance)")
    tier: Literal["flash", "pro"] = Field("flash", description="算力分級：flash / pro")


class GenerateCastRequest(BaseModel):
    event_timeline: List[str] = Field(..., description="事件時間軸陣列")
    pro_count: int = Field(2, description="高階決策主角 (Pro) 人數")
    flash_count: int = Field(10, description="基層配角/群眾 (Flash) 人數")


class SimulateAsyncRequest(BaseModel):
    event_timeline: List[str] = Field(..., description="事件時間軸陣列")
    agents: List[AgentRole] = Field(..., description="動態角色列表")
    enterprise_context: str = Field(..., description="企業內部知識/法規限制 (RAG)")
    webhook_url: str = Field(..., description="推演完成後的 Webhook 推播網址")


class TaskResponse(BaseModel):
    task_id: str = Field(..., description="非同步任務 ID")
    message: str = Field(..., description="系統狀態提示")


# ---------------------------------------------------------------------------
# 3. 核心引擎 (基於新版 google-genai)
# ---------------------------------------------------------------------------
MAX_CONCURRENT_REQUESTS = 15
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
REDUCE_CHUNK_SIZE = 20
REDIS_TTL_SECONDS = 259200  # CTO防呆：結果只保留 72 小時避免塞爆空間


async def generate_roster_via_ai(
    timeline: List[str], pro_count: int, flash_count: int
) -> List[Dict]:
    """Phase 1: 利用 Gemini 生成角色陣列 (新版寫法)"""
    formatted_timeline = "\n".join([f"- {event}" for event in timeline])
    system_prompt = """你是一個專業的商業沙盤『選角導演』。
    請嚴格回傳 JSON 陣列 (Array)，包含三個 key："role_name", "stance", "tier" ("pro" 或 "flash")。"""
    user_prompt = f"事件時間軸：\n{formatted_timeline}\n\n生成 {pro_count} 個 tier='pro' 與 {flash_count} 個 tier='flash' 的角色。確保立場多元。"

    try:
        response = await gemini_client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt, response_mime_type="application/json"
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        logger.error(f"🚨 選角生成失敗: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI 選角失敗: {str(e)}")


async def call_gemini_agent(
    agent: AgentRole, timeline: List[str], context: str
) -> dict:
    """單一角色推演 (受 Semaphore 保護)"""
    model_name = "gemini-2.5-pro" if agent.tier == "pro" else "gemini-2.5-flash"
    formatted_timeline = "\n".join([f"- {event}" for event in timeline])
    system_prompt = f"你是【{agent.role_name}】。立場：【{agent.stance}】。法規限制：{context}。拒絕廢話，直指核心。"
    user_prompt = (
        f"事件時間軸：\n{formatted_timeline}\n\n請基於你的立場給出具體反應（條列式）。"
    )

    async with semaphore:
        try:
            response = await gemini_client.aio.models.generate_content(
                model=model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(system_instruction=system_prompt),
            )
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
    """Map-Reduce 壓縮基層意見"""
    formatted_timeline = "\n".join([f"- {event}" for event in timeline])
    raw_text = "\n".join(
        [f"【{r['role']}】: {r.get('action', '失敗')}" for r in chunk_reports]
    )
    prompt = f"事件時間軸：\n{formatted_timeline}\n\n第 {chunk_id} 批反應：\n{raw_text}\n\n請濃縮成300字重點。"

    async with semaphore:
        try:
            response = await gemini_client.aio.models.generate_content(
                model="gemini-2.5-flash", contents=prompt
            )
            return f"【秘書匯報 - 批次 {chunk_id}】:\n{response.text}"
        except:
            return f"【秘書匯報失敗 - 批次 {chunk_id}】"


async def call_gemini_pro_summary(
    timeline: List[str], final_context: str, enterprise_context: str
) -> str:
    """Pro 廠長最終決策"""
    formatted_timeline = "\n".join([f"- {event}" for event in timeline])
    system_prompt = f"你是最高決策總監。根據情報統整出【最終戰略 SOP 表格】。嚴格遵守限制：{enterprise_context}"
    user_prompt = (
        f"時間軸：\n{formatted_timeline}\n\n情報：\n{final_context}\n\n請輸出報表。"
    )
    try:
        response = await gemini_client.aio.models.generate_content(
            model="gemini-2.5-pro",
            contents=user_prompt,
            config=types.GenerateContentConfig(system_instruction=system_prompt),
        )
        return response.text
    except Exception as e:
        return f"統整失敗: {str(e)}"


async def audit_decision_with_flash(decision: str, enterprise_context: str) -> str:
    """Flash 稽核防線"""
    prompt = f"你是稽核員。企業限制：{enterprise_context}\n\n待審核：\n{decision}\n\n檢查是否違規？回覆：[合規/違規] + 理由。"
    try:
        response = await gemini_client.aio.models.generate_content(
            model="gemini-2.5-flash", contents=prompt
        )
        return response.text
    except:
        return "[稽核失敗]"


# ================= 🚀 核心工作流 (Redis + Webhook) =================
async def run_simulation_pipeline_bg(task_id: str, payload: SimulateAsyncRequest):
    """執行推演，寫入 Redis，並推播 Webhook"""
    start_time = time.time()
    try:
        # 推演邏輯
        tasks = [
            call_gemini_agent(agent, payload.event_timeline, payload.enterprise_context)
            for agent in payload.agents
        ]
        all_reports = await asyncio.gather(*tasks)

        pro_reports = [r for r in all_reports if r.get("tier") == "pro"]
        flash_reports = [r for r in all_reports if r.get("tier") == "flash"]

        final_context_builder = "=== 主角 (Pro) ===\n" + "".join(
            [f"【{r['role']}】:\n{r.get('action', '')}\n---\n" for r in pro_reports]
        )
        final_context_builder += "=== 群眾 (Flash) ===\n"

        map_reduce_triggered = len(flash_reports) > REDUCE_CHUNK_SIZE
        if map_reduce_triggered:
            chunks = [
                flash_reports[i : i + REDUCE_CHUNK_SIZE]
                for i in range(0, len(flash_reports), REDUCE_CHUNK_SIZE)
            ]
            reduce_tasks = [
                reduce_flash_reports(chunk, payload.event_timeline, i + 1)
                for i, chunk in enumerate(chunks)
            ]
            summarized_chunks = await asyncio.gather(*reduce_tasks)
            final_context_builder += "\n\n".join(summarized_chunks)
        else:
            final_context_builder += "".join(
                [
                    f"【{r['role']}】:\n{r.get('action', '')}\n---\n"
                    for r in flash_reports
                ]
            )

        executive_summary = await call_gemini_pro_summary(
            payload.event_timeline, final_context_builder, payload.enterprise_context
        )
        audit_result = await audit_decision_with_flash(
            executive_summary, payload.enterprise_context
        )

        # 👑 雙保險 1：寫入 Redis 緩存 (存活 72 小時)
        result_payload = {
            "task_id": task_id,
            "status": "success",
            "execution_time_sec": round(time.time() - start_time, 2),
            "map_reduce_triggered": map_reduce_triggered,
            "audit_status": audit_result,
            "executive_summary": executive_summary,
            "agent_reports": all_reports,
        }
        await redis_client.set(
            task_id, json.dumps(result_payload), ex=REDIS_TTL_SECONDS
        )

        # 👑 雙保險 2：推播 Webhook
        async with httpx.AsyncClient() as client:
            await client.post(payload.webhook_url, json=result_payload, timeout=15.0)

    except Exception as e:
        logger.error(f"Task {task_id} 崩潰: {str(e)}")
        error_payload = {"task_id": task_id, "status": "error", "message": str(e)}
        await redis_client.set(task_id, json.dumps(error_payload), ex=REDIS_TTL_SECONDS)
        try:
            async with httpx.AsyncClient() as client:
                await client.post(payload.webhook_url, json=error_payload, timeout=10.0)
        except:
            pass


# ---------------------------------------------------------------------------
# 4. API 端點路由 (Endpoints)
# ---------------------------------------------------------------------------
@app.get("/", summary="系統狀態")
async def health_check():
    return {"status": "online", "version": "4.3.0 (Redis + GenAI SDK)"}


@app.post("/generate_cast", summary="Phase 1: AI 選角", response_model=List[AgentRole])
async def auto_generate_cast(request: GenerateCastRequest):
    return await generate_roster_via_ai(
        request.event_timeline, request.pro_count, request.flash_count
    )


@app.post("/simulate", response_model=TaskResponse, summary="Phase 2: 啟動推演")
async def run_universal_simulation(
    request: SimulateAsyncRequest, background_tasks: BackgroundTasks
):
    if not request.webhook_url.startswith("http"):
        raise HTTPException(status_code=400, detail="無效的 Webhook URL")

    task_id = f"mirofish_{int(time.time())}"

    # 啟動時先在 Redis 寫入「處理中」標記
    await redis_client.set(
        task_id,
        json.dumps({"task_id": task_id, "status": "processing"}),
        ex=REDIS_TTL_SECONDS,
    )

    background_tasks.add_task(run_simulation_pipeline_bg, task_id, request)
    return TaskResponse(task_id=task_id, message="🟢 推演啟動！任務已交由背景處理。")


# 👑 容錯端點：讓外部系統主動抓取備份資料
@app.get("/task/{task_id}", summary="Phase 3: 主動查詢推演結果 (容錯機制)")
async def get_task_result(task_id: str):
    """如果 Webhook 斷線，可憑 Task ID 來此撈取結果。資料保留 72 小時。"""
    data = await redis_client.get(task_id)
    if not data:
        raise HTTPException(status_code=404, detail="找不到該任務或資料已過期銷毀。")
    return json.loads(data)
