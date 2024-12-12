
setup:
	pip install -r requirements.txt

check:
	ruff format
	ruff check --fix