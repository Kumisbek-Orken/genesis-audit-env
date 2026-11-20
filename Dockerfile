FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

CMD ["sh", "-c", "python3 -m pytest -q && python3 scripts/generate_data.py --out-dir /tmp/eval_data --num-episodes 10 --seed 2026 --num-errors 3 && python3 scripts/run_eval.py --episodes-dir /tmp/eval_data --out /tmp/eval_data/trajectories.jsonl && rm -rf /tmp/eval_data"]
