FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY requirements.txt requirements-local.txt ./
# INSTALL_LOCAL=1 adds sentence-transformers (cross-encoder reranker); it pulls in torch (~GBs).
ARG INSTALL_LOCAL=1
RUN pip install -r requirements.txt && if [ "$INSTALL_LOCAL" = "1" ]; then pip install -r requirements-local.txt; fi
COPY app ./app
COPY ui ./ui
COPY configs ./configs
COPY scripts ./scripts
COPY eval ./eval
EXPOSE 8000 8501
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
