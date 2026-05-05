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

app = FastAPI(title="MiroFish Universal Simulation Engine", version="3.0")


# ---------------------------------------------------------------------------
# 2. 泛用型資料防呆模型 (Dynamic Pydantic Models)
# ---------------------------------------------------------------------------
class AgentRole(BaseModel):
    role_name: str = Field(
        ..., description="角色名稱，如：憤怒的網民、資深財經分析師、公關總監"
    )
    stance: str = Field(..., description="角色的基本立場或背景設定")


class SimulateRequest(BaseModel):
    client_case_description: str = Field(..., description="客戶的案件或事件描述")
    agents: List[AgentRole] = Field(..., description="本次推演需要動態生成的角色列表")


class SimulateResponse(BaseModel):
    status: str
    client_case: str
    executive_summary: str
    agent_reports: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# 3. 併發防護與動態大腦呼叫 (Anti-OOM & Fallback Core)
# ---------------------------------------------------------------------------
# 限制最高併發量，保護 Zeabur 2GB 記憶體與 Gemini 429 限制
MAX_CONCURRENT_REQUESTS = 10
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)


async def call_gemini_flash(role: AgentRole, case_desc: str) -> dict:
    """呼叫 Flash 進行基層 Agent 推演 (動態提示詞注入)"""
    model_name = "gemini-2.5-flash"

    # 👑 泛用型底層提示詞 (Universal System Prompt)
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
    system_prompt = "你是本案的最高決策總監 (CTO/CEO級別)。請根據基層 Agent 的推演回報，統整出一份【最終戰略 SOP 表格】與【風險評估】。"
    user_prompt = f"客戶案件：{case_desc}\n\n各方 Agent 推演回報如下：\n{context}\n\n請以宏觀視角輸出最終決策報表。"

    try:
        model = genai.GenerativeModel(
            model_name=model_name, system_instruction=system_prompt
        )
        response = await model.generate_content_async(user_prompt)
        return response.text
    except Exception as e:
        logger.error(f"🚨 總監決策失敗: {str(e)}")
        return "統整失敗，請檢查系統日誌。"


# ---------------------------------------------------------------------------
# 4. API 端點路由 (API Endpoints)
# ---------------------------------------------------------------------------
@app.get("/")
async def health_check():
    return {
        "status": "online",
        "message": "MiroFish Universal API is running.",
        "version": "3.0",
    }


@app.post("/simulate", response_model=SimulateResponse)
async def run_universal_simulation(request: SimulateRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="未設定 GEMINI_API_KEY 環境變數")

    # 第一階段：動態派發 Flash Agent 併發推演
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

    # 第二階段：Pro 模型大腦決策
    executive_summary = await call_gemini_pro_summary(
        request.client_case_description, compiled_reports_text
    )

    return SimulateResponse(
        status="success",
        client_case=request.client_case_description,
        executive_summary=executive_summary,
        agent_reports=base_reports,
    )
