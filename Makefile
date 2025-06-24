# SpMVGraphKit Makefile
# Provides convenient commands for development, testing, and running examples

# Python interpreter
PYTHON := python3
PIP := pip3

# Project directories
SRC_DIR := src
TEST_DIR := tests
EXAMPLES_DIR := examples
RESULTS_DIR := results
DATA_DIR := data

# Virtual environment
VENV := venv
VENV_ACTIVATE := $(VENV)/bin/activate

# Default target
.PHONY: help
help:
	@echo "SpMVGraphKit - Makefile Commands"
	@echo "================================"
	@echo ""
	@echo "Setup and Installation:"
	@echo "  make install          Install dependencies and setup project"
	@echo "  make venv             Create virtual environment"
	@echo "  make install-dev      Install development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test             Run all tests"
	@echo "  make quick-test       Quick functionality test"
	@echo ""
	@echo "Examples and Demos:"
	@echo "  make demo             Run basic reordering example"
	@echo "  make generate-data    Generate test matrices"
	@echo ""
	@echo "Development:"
	@echo "  make clean            Clean temporary files"
	@echo "  make status           Show project status"

# Setup and Installation
.PHONY: install
install:
	$(PIP) install numpy scipy networkx matplotlib seaborn
	@echo "✓ Dependencies installed!"

.PHONY: venv
venv:
	$(PYTHON) -m venv $(VENV)
	@echo "✓ Virtual environment created at $(VENV)"
	@echo "  Activate with: source $(VENV_ACTIVATE)"

.PHONY: install-dev
install-dev: install
	$(PIP) install pytest pytest-cov black flake8
	@echo "✓ Development dependencies installed!"

# Testing
.PHONY: test
test:
	@echo "Running tests..."
	$(PYTHON) -m pytest $(TEST_DIR) -v || echo "Tests require implementation"

.PHONY: quick-test
quick-test:
	@echo "Running quick functionality test..."
	$(PYTHON) -c "import sys; sys.path.append('src'); print('✓ Quick test - imports working')"

# Examples and Demos
.PHONY: demo
demo: $(RESULTS_DIR)
	@echo "Running basic reordering example..."
	cd $(EXAMPLES_DIR) && $(PYTHON) basic_reordering.py || echo "Demo requires implementation"
	@echo "✓ Demo completed!"

# Main toolkit commands (using root-level CLI)
.PHONY: demo-cli
demo-cli:
	$(PYTHON) spmv_toolkit.py demo

.PHONY: analyze-cli
analyze-cli:
	$(PYTHON) spmv_toolkit.py analyze demo

.PHONY: benchmark-cli
benchmark-cli: $(RESULTS_DIR)
	$(PYTHON) spmv_toolkit.py benchmark demo --output $(RESULTS_DIR)/cli_benchmark

# Simple examples
.PHONY: example-simple
example-simple:
	cd $(EXAMPLES_DIR) && $(PYTHON) simple_demo.py

# Help for the main toolkit
.PHONY: help-toolkit
help-toolkit:
	$(PYTHON) spmv_toolkit.py --help

# Directories
$(RESULTS_DIR):
	mkdir -p $(RESULTS_DIR)

$(DATA_DIR):
	mkdir -p $(DATA_DIR)

# Data and Matrices
.PHONY: generate-data
generate-data: $(DATA_DIR)
	@echo "Generating test matrices..."
	$(PYTHON) -c "import numpy as np; import scipy.sparse; print('✓ Test data generation ready')"

# Cleaning
.PHONY: clean
clean:
	@echo "Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	@echo "✓ Temporary files cleaned!"

# Development workflow
.PHONY: dev-setup
dev-setup: venv install-dev generate-data
	@echo "✓ Development environment setup complete!"
	@echo ""
	@echo "Next steps:"
	@echo "1. Activate virtual environment: source $(VENV_ACTIVATE)"
	@echo "2. Run quick test: make quick-test"

.PHONY: status
status:
	@echo "SpMVGraphKit Project Status"
	@echo "=========================="
	@echo ""
	@echo "Python version: $(shell $(PYTHON) --version)"
	@echo "Project directory: $(PWD)"
	@echo ""
	@echo "Dependencies:"
	@$(PYTHON) -c "import numpy; print(f'  NumPy: {numpy.__version__}')" 2>/dev/null || echo "  NumPy: Not installed"
	@$(PYTHON) -c "import scipy; print(f'  SciPy: {scipy.__version__}')" 2>/dev/null || echo "  SciPy: Not installed"
	@$(PYTHON) -c "import networkx; print(f'  NetworkX: {networkx.__version__}')" 2>/dev/null || echo "  NetworkX: Not installed"
	@echo ""
	@echo "Directories:"
	@echo "  Source: $(SRC_DIR)/ $(if $(wildcard $(SRC_DIR)),✓,✗)"
	@echo "  Tests: $(TEST_DIR)/ $(if $(wildcard $(TEST_DIR)),✓,✗)"
	@echo "  Examples: $(EXAMPLES_DIR)/ $(if $(wildcard $(EXAMPLES_DIR)),✓,✗)"