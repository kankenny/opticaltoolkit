lint:
	isort .
	flake8 .
	black .

test:
	pytest .

all: lint test
