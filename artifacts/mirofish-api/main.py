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

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("🚨 致命錯誤：未檢測到 GEMINI_API_KEY。")
else:
    genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(
    title="MiroFish Universal Engine (中英雙語操作版)",
    description="""
    **MiroFish 泛用型多智能體推演引擎 API (Universal Multi-Agent Simulation API)** 

    提供竹東傳產、公關危機、市場分析等多情境的沙盤推演能力。
    Supports multi-scenario simulations including traditional industries, PR crises, and market analysis.
    """,
    version="3.2.0",
)


# ---------------------------------------------------------------------------
# 2. 雙語泛用型資料防呆模型 (帶有強迫 UI 渲染範本)
# ---------------------------------------------------------------------------
class AgentRole(BaseModel):
    role_name: str = Field(..., description="角色名稱 (Role Name)")
    stance: str = Field(
        ..., description="角色的基本立場或背景設定 (Stance or Background)"
    )


class SimulateRequest(BaseModel):
    client_case_description: str = Field(
        ..., description="客戶的案件或事件描述 (Client Case Description)"
    )
    agents: List[AgentRole] = Field(..., description="動態角色列表 (List of Agents)")

    # 👑 CTO 魔法：強制在 Swagger UI 黑框中顯示這組雙語範本，取代預設的 "string"
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "client_case_description": "【請輸入案件/Enter Case】範例：竹東農會中秋節 2000 份急件，三天內需交貨。",
                    "agents": [
                        {
                            "role_name": "【請輸入角色/Enter Role】範例：印刷廠廠長 (Factory Manager)",
                            "stance": "【請輸入立場/Enter Stance】範例：關注產能是否能負荷，並擔心加班費超標。",
                        },
                        {
                            "role_name": "【請輸入角色/Enter Role】範例：第一線機台操作員 (Machine Operator)",
                            "stance": "【請輸入立場/Enter Stance】範例：只在乎機台會不會過熱，以及能不能準時下班。",
                        },
                    ],
                }
            ]
        }
    }


class SimulateResponse(BaseModel):
    status: str
    client_case: str
    executive_summary: str
    agent_reports: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# 3. 併發防護與動態大腦呼叫 (Anti-OOM & Fallback Core)
# ---------------------------------------------------------------------------
MAX_CONCURRENT_REQUESTS = 10
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)


async def call_gemini_flash(role: AgentRole, case_desc: str) -> dict:
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
# 4. API 端點路由
# ---------------------------------------------------------------------------
@app.get("/", summary="系統狀態 (Health Check)")
async def health_check():
    return {"status": "online", "version": "3.2.0"}


@app.post(
    "/simulate", response_model=SimulateResponse, summary="執行推演 (Run Simulation)"
)
async def run_universal_simulation(request: SimulateRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Missing GEMINI_API_KEY")

    tasks = [
        call_gemini_flash(agent, request.client_case_description)
        for agent in request.agents
    ]
    base_reports = await asyncio.gather(*tasks)

    compiled_reports_text = ""
    for report in base_reports:
        if "error" not in report:
            compiled_reports_text += f"【{report['role']}】:\n{report['action']}\n---\n"

    executive_summary = await call_gemini_pro_summary(
        request.client_case_description, compiled_reports_text
    )

    return SimulateResponse(
        status="success",
        client_case=request.client_case_description,
        executive_summary=executive_summary,
        agent_reports=base_reports,
    )
