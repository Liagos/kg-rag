FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

WORKDIR /app

# copy everything needed for package install first
COPY pyproject.toml ./
COPY uv.lock ./
COPY README.md ./
COPY src/ ./src/

# now install — src/ and README.md exist
RUN uv sync

# copy remaining files
COPY data/ ./data/
COPY entrypoint.sh ./entrypoint.sh
RUN chmod +x entrypoint.sh

ENV PYTHONPATH=/app/src

EXPOSE 8501

ENTRYPOINT ["./entrypoint.sh"]