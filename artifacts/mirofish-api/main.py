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
# 設置日誌，方便在 Zeabur 後台抓錯
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 🚨 資安防護：強制從環境變數讀取，嚴禁硬編碼 (Hardcode)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error(
        "🚨 致命錯誤：未檢測到 GEMINI_API_KEY 環境變數。請至 Zeabur 後台配置。"
    )
    # 這裡不直接 crash，讓 API 活著回傳錯誤訊息給輝哥
else:
    genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(title="MiroFish Core API (Zhudong Edition)", version="2.0")


# ---------------------------------------------------------------------------
# 2. 嚴格資料防呆模型 (Pydantic Models)
# ---------------------------------------------------------------------------
class SimulateRequest(BaseModel):
    event_description: str = Field(
        ..., description="要推演的任務描述，例如：竹東農會 2000 份客製化燙金信封案"
    )


class SimulateResponse(BaseModel):
    status: str
    task: str
    factory_manager_sop: str
    agent_reports: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# 3. Agent 實體定義與 Token 經濟學 (Agent Registry)
# ---------------------------------------------------------------------------
# 定義竹東印刷廠的標準 Agent 人設
AGENTS = [
    {
        "agent_id": "agent_prepress_01",
        "role": "印前審稿員",
        "model_tier": "gemini-2.5-flash",  # 🟢 低成本模型
        "system_prompt": "你是印刷廠的印前審稿員。收到任務後，請列出你需要檢查的檔案格式（如 CMYK、出血尺寸、解析度）與潛在客訴風險。講重點，條列式回覆。",
    },
    {
        "agent_id": "agent_machine_02",
        "role": "機台操作員",
        "model_tier": "gemini-2.5-flash",  # 🟢 低成本模型
        "system_prompt": "你是印刷廠的機台操作員。請根據任務評估：需要哪種紙材、油墨損耗預估、以及預計上機與乾燥時間。講重點，條列式回覆。",
    },
]

MANAGER_AGENT = {
    "agent_id": "agent_manager_boss",
    "role": "廠長",
    "model_tier": "gemini-2.5-pro",  # 🔴 高階決策模型
    "system_prompt": "你是竹東印刷廠的廠長。請根據各基層員工的回報，統整出一份完整的【生產 SOP 表格】。要求：必須包含步驟、負責人、注意事項。拒絕廢話，直接給表格。",
}

# ---------------------------------------------------------------------------
# 4. 併發壓力防護與大腦呼叫邏輯 (Anti-OOM & Fallback Core)
# ---------------------------------------------------------------------------
# 🚨 併發防護鎖：同一時間最多只允許 10 個協程去敲 Gemini API，保護 Zeabur 2GB 記憶體
MAX_CONCURRENT_REQUESTS = 10
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)


async def call_gemini(agent_info: dict, task_desc: str, context: str = "") -> dict:
    """封裝呼叫 Gemini 的邏輯，內建 Semaphore 鎖與自動降級 (Fallback)"""
    model_name = agent_info["model_tier"]
    role = agent_info["role"]
    prompt = f"任務：{task_desc}\n{context}"

    async with semaphore:  # 進入防護閘門
        try:
            logger.info(f"啟動 Agent [{role}] - 使用模型: {model_name}")
            model = genai.GenerativeModel(
                model_name=model_name, system_instruction=agent_info["system_prompt"]
            )
            # 異步呼叫 Gemini
            response = await model.generate_content_async(prompt)
            return {
                "agent_id": agent_info["agent_id"],
                "role": role,
                "model_used": model_name,
                "action": response.text,
            }
        except Exception as e:
            logger.error(f"🚨 Agent [{role}] 模型 {model_name} 呼叫失敗: {str(e)}")

            # 🛡️ 容錯降級機制 (Fallback)：如果 Pro 失敗，強制用 Flash 重試
            if model_name == "gemini-2.5-pro":
                logger.warning(
                    f"🔄 觸發 Fallback：Agent [{role}] 降級至 gemini-2.5-flash"
                )
                try:
                    fallback_model = genai.GenerativeModel(
                        model_name="gemini-2.5-flash",
                        system_instruction=agent_info["system_prompt"],
                    )
                    fallback_response = await fallback_model.generate_content_async(
                        prompt
                    )
                    return {
                        "agent_id": agent_info["agent_id"],
                        "role": role,
                        "model_used": "gemini-2.5-flash (Fallback)",
                        "action": fallback_response.text,
                    }
                except Exception as fallback_e:
                    return {
                        "agent_id": agent_info["agent_id"],
                        "role": role,
                        "error": str(fallback_e),
                    }

            return {"agent_id": agent_info["agent_id"], "role": role, "error": str(e)}


# ---------------------------------------------------------------------------
# 5. API 端點路由 (API Endpoints)
# ---------------------------------------------------------------------------
@app.get("/")
async def health_check():
    """系統狀態檢查，確認伺服器存活"""
    return {
        "status": "online",
        "message": "MiroFish Core API is running on Zeabur.",
        "version": "2.0",
    }


@app.post("/simulate", response_model=SimulateResponse)
async def run_simulation(request: SimulateRequest):
    """觸發多智能體推演流水線"""
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="未設定 GEMINI_API_KEY 環境變數")

    task = request.event_description

    # 第一階段：基層 Agent 同時平行推演 (Flash 模型)
    # 利用 asyncio.gather 達成異步併發，速度極快
    tasks = [call_gemini(agent, task) for agent in AGENTS]
    base_reports = await asyncio.gather(*tasks)

    # 彙整基層報告
    compiled_reports_text = "以下是基層員工的回報：\n"
    for report in base_reports:
        if "error" not in report:
            compiled_reports_text += f"【{report['role']}】:\n{report['action']}\n---\n"
        else:
            compiled_reports_text += f"【{report['role']}】回報失敗。\n---\n"

    # 第二階段：廠長進行最終決策與 SOP 生成 (Pro 模型)
    manager_report = await call_gemini(
        MANAGER_AGENT, task, context=compiled_reports_text
    )

    # 回傳純 JSON，完全解耦前端
    return SimulateResponse(
        status="success",
        task=task,
        factory_manager_sop=manager_report.get("action", "廠長決策失敗"),
        agent_reports=base_reports,
    )
