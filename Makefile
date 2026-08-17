PORT ?= $(or $(VAANI_PORT),8765)

VENV ?= .venv
PY  = $(VENV)/bin/python

install:
	python3.11 -m pip install -r requirements.txt

models:
	$(PY) scripts/download_models.py

dev:
	$(VENV)/bin/uvicorn vaani.main:app --host 0.0.0.0 --port $(PORT) --reload

dev-local:
	VAANI_SETTINGS=config/settings.local.yaml $(VENV)/bin/uvicorn vaani.main:app --host 0.0.0.0 --port $(PORT) --reload

demo-voice:
	mkdir -p demo_audio
	say -v Samantha "Namaste, main aap se appointment ke baare mein baat karna chahti hoon" -o demo_audio/user_question.aiff
	ffmpeg -y -v error -i demo_audio/user_question.aiff -ac 1 -ar 16000 -sample_fmt s16 demo_audio/user_question.wav

demo:
	$(PY) scripts/test_call.py --wav demo_audio/user_question.wav --out demo_audio/agent_reply.wav

test:
	$(PY) -m pytest -q

brain:
	docker compose -f deploy/docker-compose.brain.yml up -d

edge:
	docker compose -f deploy/docker-compose.edge.yml up --build -d

warm:
	$(PY) scripts/warm_kv_cache.py --base-url http://localhost:8000/v1