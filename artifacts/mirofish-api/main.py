import os
import json
import asyncio
import logging
import time
import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks, APIRouter
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
    title="MiroFish Universal Engine (v4.4.0 VIP 尊榮版)",
    description="實裝 VIP CSO 沉浸式對話艙與事件重塑引擎 (Gemini 2.5 核心)。",
    version="4.4.0",
)

from fastapi.middleware.cors import CORSMiddleware

# 👑 CTO 資安設定：打通跨域防線 (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# 2. 資料模型 (Schemas) - 包含新增的 VIP 專屬模型
# ---------------------------------------------------------------------------
class AgentRole(BaseModel):
    role_name: str = Field(..., description="角色名稱")
    stance: str = Field(..., description="立場設定")
    tier: Literal["flash", "pro"] = Field("flash")


class GenerateCastRequest(BaseModel):
    event_timeline: List[str]
    pro_count: int = 2
    flash_count: int = 10


class SimulateAsyncRequest(BaseModel):
    event_timeline: List[str]
    agents: List[AgentRole]
    enterprise_context: str
    webhook_url: str


class TaskResponse(BaseModel):
    task_id: str
    message: str


# 💎 VIP 專屬對話模型
class ConsultRequest(BaseModel):
    domain: str
    rules: List[str]
    initial_event: str


class SynthesizeRequest(BaseModel):
    domain: str
    initial_event: str
    cso_questions: List[Dict]
    vip_answers: str


# ---------------------------------------------------------------------------
# 3. 核心引擎 (Core Engines)
# ---------------------------------------------------------------------------
MAX_CONCURRENT_REQUESTS = 15
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
REDUCE_CHUNK_SIZE = 20
REDIS_TTL_SECONDS = 259200  # 72 小時


async def generate_roster_via_ai(
    timeline: List[str], pro_count: int, flash_count: int
) -> List[Dict]:
    formatted_timeline = "\n".join([f"- {event}" for event in timeline])
    system_prompt = "你是一個專業的商業沙盤『選角導演』。請嚴格回傳 JSON 陣列，包含三個 key：'role_name', 'stance', 'tier'。"
    user_prompt = f"事件時間軸：\n{formatted_timeline}\n\n生成 {pro_count} 個 pro 與 {flash_count} 個 flash 角色。"

    try:
        response = await gemini_client.aio.models.generate_content(
            model="gemini-2.5-flash",  # 👑 已升級為最新 2.5 Flash
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt, response_mime_type="application/json"
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        logger.error(f"🚨 選角失敗: {str(e)}")
        raise HTTPException(status_code=500, detail="AI 選角失敗")


async def call_gemini_agent(
    agent: AgentRole, timeline: List[str], context: str
) -> dict:
    # 👑 已升級為最新 2.5 Pro 與 Flash
    model_name = "gemini-2.5-pro" if agent.tier == "pro" else "gemini-2.5-flash"

    formatted_timeline = "\n".join([f"- {event}" for event in timeline])
    system_prompt = (
        f"你是【{agent.role_name}】。立場：【{agent.stance}】。法規：{context}。"
    )
    user_prompt = f"事件：\n{formatted_timeline}\n\n請基於立場給出具體反應（條列式）。"

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


async def call_gemini_pro_summary(
    timeline: List[str], final_context: str, enterprise_context: str
) -> str:
    system_prompt = f"你是最高決策總監。根據情報統整出【最終戰略 SOP 表格】。限制：{enterprise_context}"
    user_prompt = f"時間軸：\n{timeline}\n\n情報：\n{final_context}\n\n請輸出報表。"
    try:
        response = await gemini_client.aio.models.generate_content(
            model="gemini-2.5-pro",  # 👑 已升級為最新 2.5 Pro
            contents=user_prompt,
            config=types.GenerateContentConfig(system_instruction=system_prompt),
        )
        return response.text
    except Exception as e:
        return f"統整失敗: {str(e)}"


async def audit_decision_with_flash(decision: str, enterprise_context: str) -> str:
    prompt = f"你是稽核員。企業限制：{enterprise_context}\n\n待審核：\n{decision}\n\n回覆：[合規/違規] + 理由。"
    try:
        response = await gemini_client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,  # 👑 已升級為最新 2.5 Flash
        )
        return response.text
    except:
        return "[稽核失敗]"


# ================= 🚀 VIP 專屬 API 端點 (尊榮引導機制) =================


@app.post("/vip/consult", summary="VIP 尊榮幕僚質詢")
async def vip_consult(req: ConsultRequest):
    """
    CSO 首席幕僚引導：
    針對 VIP 客戶的初步描述，利用 Gemini Pro 抓出 3 個關鍵戰略盲區進行尊榮提問。
    """
    try:
        system_instruction = """
        你現在是 MiroFish 的「首席危機幕僚 (CSO)」。
        服務對象是面臨千萬美元級別危機的 VIP 董事長。

        【任務】
        1. 先用一句話專業地總結危機的嚴重性。
        2. 抓出該事件中缺失的 3 個「最致命戰略盲區」。
        3. 語氣必須尊榮、精準、具備顧問壓迫感。尊稱對方為「董事長」。

        【輸出格式 (JSON)】
        {
          "opening_statement": "董事長您好，針對...危機，目前正處於...",
          "questions": [
            {"category": "法務/財務/供應鏈", "question": "問題內容"}
          ],
          "closing_statement": "請您指示，我們將立即重塑戰略。"
        }
        """
        user_prompt = (
            f"產業：{req.domain}\n法規：{req.rules}\n事件描述：{req.initial_event}"
        )

        response = await gemini_client.aio.models.generate_content(
            model="gemini-2.5-pro",  # 👑 已升級為最新 2.5 Pro
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        logger.error(f"VIP Consult 失敗: {str(e)}")
        raise HTTPException(status_code=500, detail="幕僚交握失敗")


@app.post("/vip/synthesize", summary="VIP 事件重塑引擎")
async def vip_synthesize(req: SynthesizeRequest):
    """
    情報編譯官：
    將董事長的簡短回覆，結合原始事件，重塑為軍事級的推演戰報。
    """
    try:
        system_instruction = """
        你現在是 MiroFish 的「頂級情報編譯官」。
        任務是將 VIP 董事長的口語回覆與初步描述，編譯成一份邏輯嚴密的《高階危機戰報 (Executive Crisis Brief)》。

        【要求】
        1. 語氣冰冷、客觀、充滿數據感。
        2. 明確標示時間壓迫性、財務底線與法規風險。
        3. 這是後續所有 AI 角色推演的唯一依據。
        4. 以 Markdown 格式輸出，約 400 字。
        """
        user_prompt = f"原始事件：{req.initial_event}\n幕僚提問：{req.cso_questions}\n董事長回覆：{req.vip_answers}"

        response = await gemini_client.aio.models.generate_content(
            model="gemini-2.5-pro",  # 👑 已升級為最新 2.5 Pro
            contents=user_prompt,
            config=types.GenerateContentConfig(system_instruction=system_instruction),
        )
        return {"synthesized_event": response.text}
    except Exception as e:
        logger.error(f"VIP Synthesize 失敗: {str(e)}")
        raise HTTPException(status_code=500, detail="事件重塑失敗")


# ================= 🚀 原有推演工作流 (Background Task) =================


async def run_simulation_pipeline_bg(task_id: str, payload: SimulateAsyncRequest):
    start_time = time.time()
    try:
        tasks = [
            call_gemini_agent(agent, payload.event_timeline, payload.enterprise_context)
            for agent in payload.agents
        ]
        all_reports = await asyncio.gather(*tasks)

        pro_reports = [r for r in all_reports if r.get("tier") == "pro"]
        flash_reports = [r for r in all_reports if r.get("tier") == "flash"]

        final_context = "=== 主角 (Pro) ===\n" + "".join(
            [f"【{r['role']}】:\n{r.get('action', '')}\n---\n" for r in pro_reports]
        )
        final_context += "=== 群眾 (Flash) ===\n" + "".join(
            [f"【{r['role']}】:\n{r.get('action', '')}\n---\n" for r in flash_reports]
        )

        executive_summary = await call_gemini_pro_summary(
            payload.event_timeline, final_context, payload.enterprise_context
        )
        audit_result = await audit_decision_with_flash(
            executive_summary, payload.enterprise_context
        )

        result_payload = {
            "task_id": task_id,
            "status": "success",
            "execution_time_sec": round(time.time() - start_time, 2),
            "audit_status": audit_result,
            "executive_summary": executive_summary,
            "agent_reports": all_reports,
        }
        await redis_client.set(
            task_id, json.dumps(result_payload), ex=REDIS_TTL_SECONDS
        )

        async with httpx.AsyncClient() as client:
            await client.post(payload.webhook_url, json=result_payload, timeout=15.0)

    except Exception as e:
        logger.error(f"Task {task_id} 崩潰: {str(e)}")
        error_payload = {"task_id": task_id, "status": "error", "message": str(e)}
        await redis_client.set(task_id, json.dumps(error_payload), ex=REDIS_TTL_SECONDS)


# ---------------------------------------------------------------------------
# 4. API 路由 (Routes)
# ---------------------------------------------------------------------------
@app.get("/")
async def health_check():
    return {"status": "online", "version": "4.4.0 (VIP Edition - Gemini 2.5)"}


@app.post("/generate_cast", response_model=List[AgentRole])
async def auto_generate_cast(request: GenerateCastRequest):
    return await generate_roster_via_ai(
        request.event_timeline, request.pro_count, request.flash_count
    )


@app.post("/simulate", response_model=TaskResponse)
async def run_universal_simulation(
    request: SimulateAsyncRequest, background_tasks: BackgroundTasks
):
    task_id = f"mirofish_{int(time.time())}"
    await redis_client.set(
        task_id,
        json.dumps({"task_id": task_id, "status": "processing"}),
        ex=REDIS_TTL_SECONDS,
    )
    background_tasks.add_task(run_simulation_pipeline_bg, task_id, request)
    return TaskResponse(task_id=task_id, message="🟢 推演啟動！任務已交由背景處理。")


@app.get("/task/{task_id}")
async def get_task_result(task_id: str):
    data = await redis_client.get(task_id)
    if not data:
        raise HTTPException(status_code=404, detail="資料已過期銷毀。")
    return json.loads(data)
