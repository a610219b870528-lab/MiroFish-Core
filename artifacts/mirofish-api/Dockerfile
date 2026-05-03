# 使用官方輕量級 Python 3.11 映像檔
FROM python:3.11-slim

# 設定容器內的工作目錄
WORKDIR /app

# 複製依賴清單並安裝
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製所有 MiroFish 原始碼到容器內
COPY . .

# 暴露 FastAPI 預設 Port
EXPOSE 8080

# 啟動 Uvicorn 伺服器
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]