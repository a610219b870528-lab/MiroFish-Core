import os
import asyncio
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import google.generativeai as genai

# ---------------------------------------------------------------------------
# 1. 系統初始化與資安防護 (System Init & Security)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 🚨 資安防護：強制從環境變數讀取
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("🚨 致命錯誤：未檢測到 GEMINI_API_KEY。")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# 👑 雙語 UI 初始化：這裡的設定會直接渲染到 Swagger UI 網頁標題與描述
app = FastAPI(
    title="MiroFish Universal Engine (中英雙語版)",
    description="""
    **MiroFish 泛用型多智能體推演引擎 API (Universal Multi-Agent Simulation API)** 

    提供竹東傳產、公關危機、市場分析等多情境的沙盤推演能力。
    Supports multi-scenario simulations including traditional industries, PR crises, and market analysis.
    """,
    version="3.1.0",
)


# ---------------------------------------------------------------------------
# 2. 雙語泛用型資料防呆模型 (Bilingual Dynamic Pydantic Models)
# ---------------------------------------------------------------------------
# 這裡的 description 會直接變成操作員在網頁上看到的中文與英文提示
class AgentRole(BaseModel):
    role_name: str = Field(
        ..., description="角色名稱 (Role Name)。範例 (Example)：憤怒的網民 / 資深廠長"
    )
    stance: str = Field(
        ..., description="角色的基本立場或背景設定 (Stance or Background Setting)。"
    )


class SimulateRequest(BaseModel):
    client_case_description: str = Field(
        ..., description="客戶的案件或事件描述 (Client Case or Event Description)。"
    )
    agents: List[AgentRole] = Field(
        ...,
        description="本次推演需要動態生成的角色列表 (List of dynamically generated agents for this simulation)。",
    )


class SimulateResponse(BaseModel):
    status: str = Field(..., description="API 執行狀態 (Execution Status)")
    client_case: str = Field(..., description="原始推演案件 (Original Client Case)")
    executive_summary: str = Field(
        ...,
        description="高階總監決策與 SOP 統整 (Executive Summary & SOP from Pro Model)",
    )
    agent_reports: List[Dict[str, Any]] = Field(
        ..., description="基層特務推演原始報告 (Raw Reports from Flash Agents)"
    )


# ---------------------------------------------------------------------------
# 3. 併發防護與動態大腦呼叫 (Anti-OOM & Fallback Core) - 核心運算邏輯不變
# ---------------------------------------------------------------------------
MAX_CONCURRENT_REQUESTS = 10
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)


async def call_gemini_flash(role: AgentRole, case_desc: str) -> dict:
    """呼叫 Flash 進行基層 Agent 推演 (動態提示詞注入)"""
    model_name = "gemini-2.5-flash"

    system_prompt = f"你現在是【{role.role_name}】。你的背景與立場是：【{role.stance}】。拒絕廢話，直指核心。"
    user_prompt = f"正在推演的事件：{case_desc}\n請基於你的立場，給出你的具體反應與行動方案（條列式）。"

    async with semaphore:
        try:
            model = genai.GenerativeModel(
                model_name=model_name, system_instruction=system_prompt
            )
            response = await model.generate_content_async(user_prompt)
            return {
                "role": role.role_name,
                "stance": role.stance,
                "model_used": model_name,
                "action": response.text,
            }
        except Exception as e:
            logger.error(f"🚨 Agent [{role.role_name}] 推演失敗: {str(e)}")
            return {"role": role.role_name, "error": str(e)}


async def call_gemini_pro_summary(case_desc: str, context: str) -> str:
    """呼叫 Pro 進行高階決策統整"""
    model_name = "gemini-2.5-pro"
    system_prompt = "你是本案的最高決策總監 (CTO/CEO級別)。請根據基層 Agent 的推演回報，統整出一份【最終戰略 SOP 表格】與【風險評估】。使用繁體中文輸出。"
    user_prompt = f"客戶案件：{case_desc}\n\n各方 Agent 推演回報如下：\n{context}\n\n請以宏觀視角輸出最終決策報表。"

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
# 4. 雙語 API 端點路由 (Bilingual API Endpoints)
# ---------------------------------------------------------------------------
@app.get(
    "/",
    summary="系統健康檢查 (Health Check)",
    description="確認伺服器是否正常運作 (Check if the server is running normally).",
)
async def health_check():
    return {
        "status": "online",
        "message": "MiroFish Universal API is running.",
        "version": "3.1.0",
    }


@app.post(
    "/simulate",
    response_model=SimulateResponse,
    summary="觸發推演流水線 (Trigger Simulation Pipeline)",
    description="輸入客戶案件與動態角色，執行多智能體平行推演並由 Pro 模型產出總結。(Input case and dynamic roles to execute simulation.)",
)
async def run_universal_simulation(request: SimulateRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="未設定 GEMINI_API_KEY 環境變數 (Environment variable GEMINI_API_KEY is missing)",
        )

    # 併發派發 Flash Agent
    tasks = [
        call_gemini_flash(agent, request.client_case_description)
        for agent in request.agents
    ]
    base_reports = await asyncio.gather(*tasks)

    # 彙整報告
    compiled_reports_text = ""
    for report in base_reports:
        if "error" not in report:
            compiled_reports_text += f"【{report['role']}】:\n{report['action']}\n---\n"

    # Pro 模型決策
    executive_summary = await call_gemini_pro_summary(
        request.client_case_description, compiled_reports_text
    )

    return SimulateResponse(
        status="success",
        client_case=request.client_case_description,
        executive_summary=executive_summary,
        agent_reports=base_reports,
    )
