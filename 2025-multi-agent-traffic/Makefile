# project variables
BASHRC_FILE := ~/.bashrc

# ========================================================
# Install
# ========================================================
# Use pyenv as package python manager
install-uv:
	curl -LsSf https://astral.sh/uv/install.sh | sh

# ========================================================
# Setup
# ========================================================
# make command to setup python venv
setup: clean-python uv-setup uv-tox pre-commit-install

uv-setup:
	uv python install 3.10 3.11 3.12
	uv python pin 3.11
	uv sync
	uv build

uv-tox:
	uv tool install tox --with tox-uv --force

pre-commit-install:
	@echo "Setting up pre-commit"
	uv tool install pre-commit --with pre-commit-uv --force
	uvx pre-commit install
	uvx pre-commit autoupdate

# ========================================================
# Code Formatting
# ========================================================
format:
	uvx ruff check --fix
	uvx ruff format

# ========================================================
# Clean
# ========================================================
# clean python venv
clean-python:
	rm -rf .venv
	rm -rf ./.git/hooks
	rm -f .python-version
	rm -f uv.lock
	rm -rf .tox/
	rm -rf build dist src.egg-info
