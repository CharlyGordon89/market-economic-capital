FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy project metadata and source
COPY pyproject.toml requirements.txt README.md ./
COPY src ./src
COPY configs ./configs

# install deps and the package (editable) so console scripts are registered
RUN pip install --upgrade pip && \
    (test -f requirements.txt && pip install -r requirements.txt || true) && \
    pip install -e .[dev]

ENTRYPOINT ["market-ec"]
CMD ["--help"]

# In your Dockerfile
RUN pip install plotly
