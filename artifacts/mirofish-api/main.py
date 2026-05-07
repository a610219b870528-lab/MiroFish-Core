import os
import json
import asyncio
import logging
import time
import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks, APIRouter
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Literal, Optional
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

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

app = FastAPI(
    title="MiroFish Universal Engine (v5.0 Oracle 預知回推版)",
    description="實裝 VIP CSO 沉浸式對話艙、多輪剝洋蔥質詢與可解釋選角 (Gemini 2.5 核心)。",
    version="5.0.0",
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# 2. 資料模型 (Schemas) - 👑 升級可解釋選角與歷史記憶
# ---------------------------------------------------------------------------
class AgentRole(BaseModel):
    role_name: str = Field(..., description="角色名稱")
    stance: str = Field(..., description="立場設定")
    tier: Literal["flash", "pro"] = Field("flash")
    # 👑 VIP 專屬：可解釋性選角理由
    justification: str = Field(default="一般沙盤推演預設配置", description="推薦理由")


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


# 💎 VIP 專屬對話與歷史記憶模型
class ChatMessage(BaseModel):
    role: Literal["cso", "vip"]
    content: str


class ConsultRequest(BaseModel):
    domain: str
    rules: List[str]
    initial_event: str
    history: List[ChatMessage] = []  # 👑 接收歷史對話，實現多輪深挖


class SynthesizeRequest(BaseModel):
    domain: str
    initial_event: str
    history: List[ChatMessage]  # 👑 基於完整對話歷史進行重塑


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
    # 👑 升級：強制要求 AI 解釋選角理由
    system_prompt = """你是一個專業的商業沙盤『選角導演』。
    請嚴格回傳 JSON 陣列，包含四個 key：'role_name', 'stance', 'tier', 'justification'。
    'justification' 必須引用事件細節，解釋為何推薦此角色（約30字以內）。"""

    user_prompt = f"事件時間/情報：\n{formatted_timeline}\n\n請生成 {pro_count} 個 pro 與 {flash_count} 個 flash 角色。"

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
        logger.error(f"🚨 選角失敗: {str(e)}")
        raise HTTPException(status_code=500, detail="AI 選角失敗")


async def call_gemini_agent(
    agent: AgentRole, timeline: List[str], context: str
) -> dict:
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
            model="gemini-2.5-pro",
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
            model="gemini-2.5-flash", contents=prompt
        )
        return response.text
    except:
        return "[稽核失敗]"


# ================= 🚀 VIP 專屬 API 端點 (預知回推機制) =================


@app.post("/vip/consult", summary="VIP 預知回推與多輪質詢")
async def vip_consult(req: ConsultRequest):
    """
    先知協定 (The Oracle Protocol)：
    基於起火點與對話歷史，先預測最壞終局，再逼問戰略底線。若情報飽和則結束對話。
    """
    try:
        system_instruction = """
        你現在是 MiroFish 的「首席危機幕僚 (CSO)」。服務對象是面臨千萬美元危機的 VIP 董事長。
        請閱讀客戶的初始情報與【歷史對話紀錄】。

        【任務邏輯】
        1. 預知回推：如果是第一次提問，請先給出一個令人背脊發涼的「最壞終局預測」(oracle_projection)，接著基於該終局，提出 2-3 個戰略盲區問題 (strategic_questions)。
        2. 剝洋蔥深挖：如果已有歷史對話，請根據董事長上一輪的回覆，繼續向下追問更細節的授權或底線。
        3. 情報收網：如果經過幾輪對話，你認為情報已經具備極高的「法務/財務/備援」戰略飽和度，請將 status 設為 "ready"。否則設為 "interrogating"。
        4. 語氣：尊榮、客觀、具備麥肯錫頂級顧問的壓迫感與專業度。

        【輸出格式 (嚴格 JSON)】
        {
          "oracle_projection": "【最壞終局預測】...", // 若非首回合，可簡略或針對新回答做次要預測
          "strategic_questions": [
            {"category": "例如: 財務彈藥庫", "question": "問題內容"}
          ],
          "status": "interrogating" // 若情報已足夠推演，請填 "ready"
        }
        """

        # 組裝歷史對話
        history_text = "無（這是第一回合）"
        if req.history:
            history_text = "\n".join(
                [
                    f"[{'董事長' if msg.role == 'vip' else '首席幕僚'}]: {msg.content}"
                    for msg in req.history
                ]
            )

        user_prompt = (
            f"產業：{req.domain}\n"
            f"法規/合約重點：{req.rules}\n"
            f"初始起火點：{req.initial_event}\n\n"
            f"【歷史對話紀錄】\n{history_text}\n\n"
            f"請給出您的下一步質詢或判斷。"
        )

        response = await gemini_client.aio.models.generate_content(
            model="gemini-2.5-pro",
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


@app.post("/vip/synthesize", summary="VIP 全盤戰報重塑")
async def vip_synthesize(req: SynthesizeRequest):
    """
    情報編譯官：將多輪對話的精華，濃縮成無懈可擊的《高階危機戰報》。
    """
    try:
        system_instruction = """
        你現在是 MiroFish 的「頂級情報編譯官」。
        請將 VIP 董事長的「初始起火點」以及「與幕僚的多輪對話歷史」，編譯成一份邏輯嚴密的《高階危機戰報 (Executive Crisis Brief)》。

        【要求】
        1. 語氣冰冷、客觀、充滿數據感。
        2. 必須在文本中明確標示出對話中確認的：財務授權底線、法務合約限制、備援與時間壓迫。
        3. 這是後續所有 AI 角色推演的唯一依據，不可遺漏決策者的任何授權。
        4. 以 Markdown 格式輸出，約 400-500 字。
        """
        history_text = "\n".join(
            [
                f"[{'董事長' if msg.role == 'vip' else '首席幕僚'}]: {msg.content}"
                for msg in req.history
            ]
        )
        user_prompt = (
            f"原始事件：{req.initial_event}\n\n【完整諮詢對話】\n{history_text}"
        )

        response = await gemini_client.aio.models.generate_content(
            model="gemini-2.5-pro",
            contents=user_prompt,
            config=types.GenerateContentConfig(system_instruction=system_instruction),
        )
        return {"synthesized_event": response.text}
    except Exception as e:
        logger.error(f"VIP Synthesize 失敗: {str(e)}")
        raise HTTPException(status_code=500, detail="戰報重塑失敗")


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
    return {"status": "online", "version": "5.0.0 (Oracle Protocol Edition)"}


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
