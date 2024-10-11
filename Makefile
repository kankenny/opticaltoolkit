setup:
	pip install -r requirements.txt

lint:
	isort .
	black .
	flake8 .

test:
	pytest .

all: lint test
