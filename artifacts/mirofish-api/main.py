import os
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
    title="MiroFish Universal Engine (百人 Map-Reduce 巨型沙盤版)",
    description="""
    **MiroFish 泛用型多智能體推演引擎 API** 

    支援「自訂總人數」與「A/B 時間軸干預」。內建 Map-Reduce 階層壓縮與 Semaphore 併發防護網，最高支援數百名 Agent 同時推演而不崩潰。
    """,
    version="4.0.0",
)


# ---------------------------------------------------------------------------
# 2. 雙語化資料模型 (支援動態算力分級 Tier)
# ---------------------------------------------------------------------------
class AgentRole(BaseModel):
    role_name: str = Field(..., description="角色名稱 (Role Name)")
    stance: str = Field(
        ..., description="角色的基本立場或背景設定 (Stance or Background)"
    )
    tier: Literal["flash", "pro"] = Field(
        "flash",
        description="大腦算力分級：flash (低成本群眾配角) / pro (高智能決策主角)",
    )


class SimulateRequest(BaseModel):
    event_timeline: List[str] = Field(
        ..., description="事件時間軸陣列 (Array of chronological events)"
    )
    agents: List[AgentRole] = Field(..., description="動態角色列表 (List of Agents)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "event_timeline": [
                        "【方案A】調漲社區管理費：每坪增加 20 元，增聘夜間雙哨保全。",
                        "【突發干預】管委會主委在群組表示若不通過將集體請辭。",
                    ],
                    "agents": [
                        {
                            "role_name": "【主角】永旭保全夜班隊長",
                            "stance": "關注弟兄安危與排班，強烈支持增聘雙哨。",
                            "tier": "pro",
                        },
                        {
                            "role_name": "【配角 1】投資客房東",
                            "stance": "拒絕任何會增加持有成本的方案。",
                            "tier": "flash",
                        },
                        {
                            "role_name": "【配角 2】年輕雙薪家庭",
                            "stance": "注重安全，只要能提出巡邏財報就願意支持。",
                            "tier": "flash",
                        },
                    ],
                }
            ]
        }
    }


class SimulateResponse(BaseModel):
    status: str = Field(..., description="API 執行狀態 (Execution Status)")
    map_reduce_triggered: bool = Field(
        ..., description="是否觸發百人壓縮機制 (Was Map-Reduce triggered?)"
    )
    executive_summary: str = Field(
        ..., description="總監決策與 SOP 統整 (Executive Summary & SOP)"
    )
    agent_reports: List[Dict[str, Any]] = Field(
        ..., description="所有特務的原始推演報告 (Raw Reports)"
    )


# ---------------------------------------------------------------------------
# 3. 併發防護與 Map-Reduce 核心引擎 (Anti-OOM Core)
# ---------------------------------------------------------------------------
# 🚨 絕對防線：一次最多只允許 15 個請求同時敲擊 Gemini，保護 Zeabur 2GB RAM
MAX_CONCURRENT_REQUESTS = 15
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
# Map-Reduce 觸發閾值：當 Flash 配角超過 20 人時啟動壓縮機制
REDUCE_CHUNK_SIZE = 20


async def call_gemini_agent(agent: AgentRole, timeline: List[str]) -> dict:
    """底層推演：依據 tier 動態切換模型 (主角 Pro / 配角 Flash)"""
    model_name = "gemini-2.5-pro" if agent.tier == "pro" else "gemini-2.5-flash"
    formatted_timeline = "\n".join([f"- {event}" for event in timeline])

    system_prompt = f"你現在是【{agent.role_name}】。你的背景與立場是：【{agent.stance}】。拒絕廢話，直指核心。"
    user_prompt = f"以下是連續發生的時間軸事件：\n{formatted_timeline}\n\n請基於你的立場給出具體反應（條列式）。"

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
            logger.error(f"🚨 Agent [{agent.role_name}] 失敗: {str(e)}")
            return {"role": agent.role_name, "tier": agent.tier, "error": str(e)}


async def reduce_flash_reports(
    chunk_reports: List[Dict], timeline: List[str], chunk_id: int
) -> str:
    """Map-Reduce 秘書壓縮層：將 20 個配角的意見濃縮"""
    model_name = "gemini-2.5-flash"
    formatted_timeline = "\n".join([f"- {event}" for event in timeline])
    raw_text = "\n".join(
        [f"【{r['role']}】: {r.get('action', '失敗')}" for r in chunk_reports]
    )

    prompt = f"事件時間軸：\n{formatted_timeline}\n\n以下是第 {chunk_id} 批基層群眾/配角的反應：\n{raw_text}\n\n請扮演『情報秘書』，將上述群眾意見濃縮成 300 字以內的情緒風向與核心訴求重點。"

    async with semaphore:
        try:
            model = genai.GenerativeModel(model_name=model_name)
            response = await model.generate_content_async(prompt)
            return f"【秘書群眾風向匯報 - 批次 {chunk_id}】:\n{response.text}"
        except Exception as e:
            return f"【秘書匯報 - 批次 {chunk_id} 失敗】: {str(e)}"


async def call_gemini_pro_summary(timeline: List[str], final_context: str) -> str:
    """終極廠長決策：匯整主角意見與秘書濃縮的群眾摘要"""
    model_name = "gemini-2.5-pro"
    formatted_timeline = "\n".join([f"- {event}" for event in timeline])

    system_prompt = "你是本案的最高決策總監。請根據『時間軸事態發展』、『核心主角意見』與『基層群眾風向』，統整出解決危機的【最終戰略 SOP 表格】。使用繁體中文輸出。"
    user_prompt = f"事態時間軸：\n{formatted_timeline}\n\n情報匯總如下：\n{final_context}\n\n請以宏觀視角輸出決策報表。"

    try:
        model = genai.GenerativeModel(
            model_name=model_name, system_instruction=system_prompt
        )
        response = await model.generate_content_async(user_prompt)
        return response.text
    except Exception as e:
        logger.error(f"🚨 總監決策失敗: {str(e)}")
        return "統整失敗 (Summary Failed)，請檢查系統日誌。"


# ---------------------------------------------------------------------------
# 4. API 端點路由 (API Endpoints)
# ---------------------------------------------------------------------------
@app.post(
    "/simulate", response_model=SimulateResponse, summary="百人級 Map-Reduce 沙盤推演"
)
async def run_universal_simulation(request: SimulateRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Missing GEMINI_API_KEY")

    # 1. 併發推演所有 Agent (Map)
    tasks = [
        call_gemini_agent(agent, request.event_timeline) for agent in request.agents
    ]
    all_reports = await asyncio.gather(*tasks)

    # 2. 分離主角 (Pro) 與配角 (Flash)
    pro_reports = [r for r in all_reports if r.get("tier") == "pro"]
    flash_reports = [r for r in all_reports if r.get("tier") == "flash"]

    final_context_builder = ""
    map_reduce_triggered = False

    # 3. 處理主角報告 (不壓縮，保留完整決策細節)
    if pro_reports:
        final_context_builder += "=== 核心主角 (Pro) 意見 ===\n"
        for r in pro_reports:
            final_context_builder += (
                f"【{r['role']}】:\n{r.get('action', '錯誤')}\n---\n"
            )

    # 4. 處理配角報告 (Reduce 壓縮機制)
    final_context_builder += "=== 基層群眾 (Flash) 風向 ===\n"
    if len(flash_reports) <= REDUCE_CHUNK_SIZE:
        # 人數少，直接呈報
        for r in flash_reports:
            final_context_builder += (
                f"【{r['role']}】:\n{r.get('action', '錯誤')}\n---\n"
            )
    else:
        # 人數超過 20，觸發 Map-Reduce 秘書壓縮
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

    # 5. 總結廠長決策
    executive_summary = await call_gemini_pro_summary(
        request.event_timeline, final_context_builder
    )

    return SimulateResponse(
        status="success",
        map_reduce_triggered=map_reduce_triggered,
        executive_summary=executive_summary,
        agent_reports=all_reports,  # 依然回傳所有生肉報告供操作員前端備查
    )
