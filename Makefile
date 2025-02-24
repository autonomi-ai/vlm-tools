.PHONY: lint

default: help;

help:
	@echo "vlm-tools: Tools to work with the VLM-1 API."
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  clean               Remove all build, test, coverage and Python artifacts"
	@echo "  clean-build         Remove build artifacts"
	@echo "  clean-pyc           Remove Python file artifacts"
	@echo "  clean-test          Remove test and coverage artifacts"
	@echo "  lint                Format source code automatically"
	@echo "  test                Basic GPU/CPU testing with a single GPU"
	@echo "  dist                Builds source and wheel package"
	@echo ""

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	rm -fr site/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +


clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

lint: ## Format source code automatically
	pre-commit run --all-files # Uses pyproject.toml

test: ## Basic tests
	pytest -sv tests

test-all: ## Basic tests + docker builds
	pytest -sv tests -sv -m 'not (skip)'

dist: clean ## builds source and wheel package
	python -m build --sdist --wheel
	ls -lh dist

create-pypi-release: dist  ## package, git tag/release and upload a release to PyPI
	twine --version
	@echo -n "Are you sure you want to create a PyPI release? [y/N] " && read ans && [ $${ans:-N} = y ]
	twine upload dist/*.whl
	echo "Successfully created release for vlm-tools."
