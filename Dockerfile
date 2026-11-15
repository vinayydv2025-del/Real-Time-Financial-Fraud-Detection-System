
FROM python:3.12-slim

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .

RUN pip install \
    --no-cache-dir \
    --default-timeout=120 \
    --retries=5 \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    -r requirements.txt

# Copy application source code
COPY api/ ./api/
COPY app/ ./app/
COPY src/ ./src/
COPY models/ ./models/
COPY reports/ ./reports/
COPY data/simulation/simulation_transactions.csv ./data/simulation/simulation_transactions.csv
# Create runtime directories
RUN mkdir -p /app/runtime /app/logs

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000 8501

# Start FastAPI internally and Streamlit on Render's public port
CMD ["sh", "-c", "uvicorn api.main:app --host 127.0.0.1 --port 8000 & exec streamlit run app/streamlit_app.py --server.address=0.0.0.0 --server.port=${PORT:-8501} --server.headless=true"]
