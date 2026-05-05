import os
import json
import asyncio
import logging
import time
import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Literal
import google.generativeai as genai

# ---------------------------------------------------------------------------
# 1. 系統初始化與資安防護 (System Init & Security)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 🚨 CTO 防線：嚴禁 API Key 硬編碼
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("🚨 致命錯誤：未檢測到 GEMINI_API_KEY。")
else:
    genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(
    title="MiroFish Universal Engine (非同步 RAG Webhook 版)",
    description="""
    **兩階段流水線 (v4.2.0)：**
    1. 使用 `/generate_cast` 讓 AI 自動生成百人角色清單。
    2. 將清單送入 `/simulate` 進行背景非同步推演，並透過 Webhook 推播結果。
    """,
    version="4.2.0",
)


# ---------------------------------------------------------------------------
# 2. 雙語化資料模型 (Bilingual Schemas)
# ---------------------------------------------------------------------------
class AgentRole(BaseModel):
    role_name: str = Field(..., description="角色名稱 (Role Name)")
    stance: str = Field(
        ..., description="角色的基本立場或背景設定 (Stance or Background)"
    )
    tier: Literal["flash", "pro"] = Field(
        "flash", description="算力分級：flash (配角) / pro (主角)"
    )


class GenerateCastRequest(BaseModel):
    event_timeline: List[str] = Field(
        ..., description="事件時間軸陣列 (Array of chronological events)"
    )
    pro_count: int = Field(2, description="需要 AI 生成幾個『高階決策主角 (Pro)』？")
    flash_count: int = Field(
        10, description="需要 AI 生成幾個『基層配角/群眾 (Flash)』？"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "event_timeline": ["印刷廠四色機突發燒毀，無法接急單。"],
                    "pro_count": 2,
                    "flash_count": 5,
                }
            ]
        }
    )


class SimulateAsyncRequest(BaseModel):
    event_timeline: List[str] = Field(
        ..., description="事件時間軸陣列 (Event Timeline)"
    )
    agents: List[AgentRole] = Field(..., description="動態角色列表 (Agent Roster)")
    enterprise_context: str = Field(
        ..., description="企業內部知識/法規限制 (Enterprise Rules/RAG)"
    )
    webhook_url: str = Field(
        ..., description="推演完成後的 Webhook 推播網址 (Webhook Delivery URL)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "event_timeline": [
                        "保全系統顯示竹科廠區有異常入侵訊號。",
                        "確認為野貓誤觸，但夜班保全已啟動封鎖。",
                    ],
                    "agents": [
                        {
                            "role_name": "中控室主任",
                            "stance": "負責全廠調度，重視SOP",
                            "tier": "pro",
                        },
                        {
                            "role_name": "夜班機動保全",
                            "stance": "第一線巡邏，疲勞值高",
                            "tier": "flash",
                        },
                    ],
                    "enterprise_context": "永旭保全法規：誤報事件需於 15 分鐘內解除封鎖，夜班保全連續走動不可超過 2 小時。",
                    "webhook_url": "https://your-webhook.com/catch",
                }
            ]
        }
    )


class TaskResponse(BaseModel):
    task_id: str = Field(..., description="非同步任務 ID")
    message: str = Field(..., description="系統狀態提示")


# ---------------------------------------------------------------------------
# 3. 核心引擎 (選角 + 推演 + Map-Reduce + 稽核)
# ---------------------------------------------------------------------------
MAX_CONCURRENT_REQUESTS = 15
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
REDUCE_CHUNK_SIZE = 20


async def generate_roster_via_ai(
    timeline: List[str], pro_count: int, flash_count: int
) -> List[Dict]:
    """利用 Gemini (Flash) 產生 JSON 角色陣列"""
    formatted_timeline = "\n".join([f"- {event}" for event in timeline])
    generation_config = genai.GenerationConfig(response_mime_type="application/json")

    system_prompt = """你是一個專業的商業沙盤『選角導演』。
    請根據傳入的事件，設計出符合衝突與推演邏輯的利害關係人。
    嚴格回傳 JSON 陣列 (Array)，包含三個 key："role_name", "stance", "tier" ("pro" 或 "flash")。"""
    user_prompt = f"事件時間軸：\n{formatted_timeline}\n\n生成 {pro_count} 個高階主角 (tier='pro') 與 {flash_count} 個基層配角 (tier='flash')。確保立場多元。"

    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=system_prompt,
            generation_config=generation_config,
        )
        response = await model.generate_content_async(user_prompt)
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

    system_prompt = f"你是【{agent.role_name}】。立場：【{agent.stance}】。企業背景與限制：{context}。拒絕廢話，直指核心。"
    user_prompt = f"事件時間軸：\n{formatted_timeline}\n\n請基於你的立場與法規限制，給出具體反應與行動（條列式）。"

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
    """Map-Reduce 壓縮基層意見"""
    formatted_timeline = "\n".join([f"- {event}" for event in timeline])
    raw_text = "\n".join(
        [f"【{r['role']}】: {r.get('action', '失敗')}" for r in chunk_reports]
    )
    prompt = f"事件時間軸：\n{formatted_timeline}\n\n第 {chunk_id} 批基層反應：\n{raw_text}\n\n請濃縮成 300 字以內的情緒風向與重點。"

    async with semaphore:
        try:
            model = genai.GenerativeModel(model_name="gemini-2.5-flash")
            response = await model.generate_content_async(prompt)
            return f"【秘書風向匯報 - 批次 {chunk_id}】:\n{response.text}"
        except:
            return f"【秘書匯報失敗 - 批次 {chunk_id}】"


async def call_gemini_pro_summary(
    timeline: List[str], final_context: str, enterprise_context: str
) -> str:
    """Pro 廠長最終決策"""
    formatted_timeline = "\n".join([f"- {event}" for event in timeline])
    system_prompt = f"你是本案最高決策總監。根據情報統整出【最終戰略 SOP 表格】。嚴格遵守此剛性限制/法規：{enterprise_context}"
    user_prompt = f"事態時間軸：\n{formatted_timeline}\n\n情報匯總：\n{final_context}\n\n請輸出決策報表。"

    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.5-pro", system_instruction=system_prompt
        )
        response = await model.generate_content_async(user_prompt)
        return response.text
    except Exception as e:
        return f"統整失敗: {str(e)}"


async def audit_decision_with_flash(decision: str, enterprise_context: str) -> str:
    """Flash 稽核防線 (Sanity Check)"""
    prompt = f"你是嚴格的法規稽核員。企業限制：{enterprise_context}\n\n待審核決策：\n{decision}\n\n請檢查決策是否違反限制？回覆格式：[合規/違規] + 簡短理由。"
    try:
        model = genai.GenerativeModel(model_name="gemini-2.5-flash")
        response = await model.generate_content_async(prompt)
        return response.text
    except:
        return "[稽核失敗] 無法驗證合規性"


# ================= 新增：背景推演工作流 (Background Task) =================
async def run_simulation_pipeline_bg(task_id: str, payload: SimulateAsyncRequest):
    """執行完整沙盤推演並發送 Webhook"""
    logger.info(f"Task {task_id} 啟動推演...")
    start_time = time.time()

    try:
        # 1. 執行角色推演
        tasks = [
            call_gemini_agent(agent, payload.event_timeline, payload.enterprise_context)
            for agent in payload.agents
        ]
        all_reports = await asyncio.gather(*tasks)

        pro_reports = [r for r in all_reports if r.get("tier") == "pro"]
        flash_reports = [r for r in all_reports if r.get("tier") == "flash"]

        # 2. 彙整與 Map-Reduce
        final_context_builder = "=== 核心主角 (Pro) 意見 ===\n"
        for r in pro_reports:
            final_context_builder += (
                f"【{r['role']}】:\n{r.get('action', '錯誤')}\n---\n"
            )

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
                reduce_flash_reports(chunk, payload.event_timeline, i + 1)
                for i, chunk in enumerate(chunks)
            ]
            summarized_chunks = await asyncio.gather(*reduce_tasks)
            final_context_builder += "\n\n".join(summarized_chunks)

        # 3. 廠長總結
        executive_summary = await call_gemini_pro_summary(
            payload.event_timeline, final_context_builder, payload.enterprise_context
        )

        # 4. Flash 稽核
        audit_result = await audit_decision_with_flash(
            executive_summary, payload.enterprise_context
        )

        # 5. 打包 Payload 準備推送
        execution_time = round(time.time() - start_time, 2)
        webhook_payload = {
            "task_id": task_id,
            "status": "success",
            "execution_time_sec": execution_time,
            "map_reduce_triggered": map_reduce_triggered,
            "audit_status": audit_result,
            "executive_summary": executive_summary,
            "agent_reports": all_reports,
        }

        # 發送 Webhook (非阻塞)
        async with httpx.AsyncClient() as client:
            await client.post(payload.webhook_url, json=webhook_payload, timeout=15.0)
            logger.info(f"Task {task_id} 推播成功！")

    except Exception as e:
        logger.error(f"Task {task_id} 崩潰: {str(e)}")
        error_payload = {"task_id": task_id, "status": "error", "message": str(e)}
        try:
            async with httpx.AsyncClient() as client:
                await client.post(payload.webhook_url, json=error_payload, timeout=10.0)
        except:
            pass


# ---------------------------------------------------------------------------
# 4. 兩階段 API 端點路由 (Endpoints)
# ---------------------------------------------------------------------------
@app.get("/", summary="系統狀態")
async def health_check():
    return {"status": "online", "version": "4.2.0 (Async + RAG Edition)"}


@app.post(
    "/generate_cast",
    summary="Phase 1: AI 選角導演 (Auto-Cast)",
    response_model=List[AgentRole],
)
async def auto_generate_cast(request: GenerateCastRequest):
    """輸入事件與需求人數，AI 自動生成利益關係人清單。 (同步端點)"""
    return await generate_roster_via_ai(
        request.event_timeline, request.pro_count, request.flash_count
    )


@app.post(
    "/simulate",
    response_model=TaskResponse,
    summary="Phase 2: 啟動非同步沙盤推演 (Async Simulation)",
)
async def run_universal_simulation(
    request: SimulateAsyncRequest, background_tasks: BackgroundTasks
):
    """
    將選角清單、法規限制送入。
    系統將立刻回傳 Task ID，並於背景執行百人推演與 Flash 稽核，完成後推播至 Webhook。
    """
    if not request.webhook_url.startswith("http"):
        raise HTTPException(status_code=400, detail="無效的 Webhook URL 格式。")

    task_id = f"mirofish_{int(time.time())}"

    # 派遣任務到背景執行
    background_tasks.add_task(run_simulation_pipeline_bg, task_id, request)

    return TaskResponse(
        task_id=task_id,
        message="🟢 請求已受理。沙盤推演與法規稽核正在背景執行，完成後將自動推播至 Webhook。",
    )
