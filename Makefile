.PHONY: setup gen all test bench qc report

setup:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -e .

gen:
	.venv/bin/python -m vtn gen --seed 7

all:
	.venv/bin/python -m vtn all --seed 7

test:
	.venv/bin/pytest -q --cov=src/vtn --cov-report=term-missing

bench:
	.venv/bin/python -m vtn bench

qc:
	.venv/bin/python -m vtn qc
