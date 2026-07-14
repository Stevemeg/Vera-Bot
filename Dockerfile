FROM python:3.11-slim

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Opt-in disk snapshot so an unplanned in-container process restart
# (supervisor crash-restart, OOM kill, etc.) doesn't lose context mid-test.
# Deleted automatically by POST /v1/teardown, so this never becomes
# post-test data retention (see app/store.py's docstring). Unset this env
# var to run purely in-memory, matching the challenge's default
# expectation that no restarts happen during a test window.
ENV VERA_PERSIST_PATH=/srv/data/context_snapshot.json
RUN mkdir -p /srv/data

EXPOSE 8080

# 1 worker: state (ContextStore/SuppressionEngine/ConversationStore/
# DecisionLog) is in-memory and per-process. Do NOT scale workers beyond 1
# unless you also move that state to a shared backend (Redis, etc.) — see
# README "What I'd add with more time".
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
