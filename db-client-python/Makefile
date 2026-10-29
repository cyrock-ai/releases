# Development entry points. CI runs the same checks; see .github/workflows/python.yml.
PYTHON ?= python3
VENV   ?= .venv
BIN     = $(VENV)/bin

.PHONY: venv proto lint typecheck test integration check build clean

# Order matters: the dev group brings grpcio-tools, generate_proto.py uses it to write the
# src/cyrock_db/_version.py that `pip install -e .` reads the dynamic version from. CI does the same.
# pip is upgraded first because `--group` needs pip >= 25.1.
venv:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --quiet --upgrade pip
	$(BIN)/pip install --quiet --group dev
	$(BIN)/python scripts/generate_proto.py
	$(BIN)/pip install --quiet -e .

# Regenerates src/cyrock_db/_proto from ../proto/src/main/proto. Needs the reactor checkout.
proto:
	$(BIN)/python scripts/generate_proto.py

# No `ruff format --check`: the formatter collapses the vertically aligned layout the house style asks
# for (see CLAUDE.md, Python section), so the aligned style wins and only the linter is enforced.
lint:
	$(BIN)/ruff check .

typecheck:
	$(BIN)/mypy

test:
	$(BIN)/pytest

# Starts a real CYROCK.AI DB via the reactor's Bootstrap runner. Needs a built reactor:
#   mvn -q install -DskipTests -Djacoco.skip=true
integration:
	CYROCK_IT=1 $(BIN)/pytest integration/ -v

check: lint typecheck test

build:
	$(BIN)/python -m build

clean:
	rm -rf dist build .pytest_cache .mypy_cache .ruff_cache src/cyrock_db/_proto/*_pb2*.py*
