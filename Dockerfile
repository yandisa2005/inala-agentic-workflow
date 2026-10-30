FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source from src/ directory
COPY src/ ./src/

# Run via module or direct path
ENTRYPOINT ["python", "src/box-agentic-workflow.py"]