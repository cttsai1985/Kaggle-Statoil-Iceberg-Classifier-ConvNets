FROM ghcr.io/astral-sh/uv:0.9.24 AS uv
FROM python:3.12-slim-bookworm AS runtime
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_LINK_MODE=copy \
    PATH=/opt/venv/bin:$PATH PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/workspace/src MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /opt/iceberg
COPY pyproject.toml uv.lock ./
COPY README.md LICENSE ./
COPY src ./src
RUN uv sync --locked --no-dev
RUN useradd --create-home --uid 1000 researcher
USER researcher
WORKDIR /workspace
ENTRYPOINT ["iceberg"]

FROM runtime AS lab
USER root
WORKDIR /opt/iceberg
RUN uv sync --locked --no-dev --group lab --group data
COPY containers/jupyter_server_config.py /etc/jupyter/jupyter_server_config.py
USER researcher
WORKDIR /workspace
ENTRYPOINT ["jupyter", "lab"]
CMD ["--config=/etc/jupyter/jupyter_server_config.py"]
