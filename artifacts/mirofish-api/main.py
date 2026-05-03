from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from agent_core import run_simulation
import uvicorn

app = FastAPI(title="MiroFish Core API", version="1.1.0")


# 👑 CTO 嚴格定義的資料接收格式 (防呆機制)
class SimulationRequest(BaseModel):
    event_description: str = Field(
        ..., description="要進行沙盤推演的突發事件或公關危機", min_length=5
    )


@app.get("/")
def read_root():
    return {
        "status": "MiroFish Core Active",
        "env": "Development (Replit)",
        "version": "1.1.0",
    }


@app.post("/simulate")
async def trigger_simulation(payload: SimulationRequest):
    """
    觸發 MiroFish 社會推演 (接收動態事件 Payload)
    """
    try:
        # 將前端/NemoClaw 傳來的事件字串，餵給核心引擎
        result = await run_simulation(event_context=payload.event_description)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推演引擎崩潰: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
