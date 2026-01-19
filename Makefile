# Makefile for embedmit R package
# Provides test automation targets for TDD workflow

.PHONY: test test-all test-comparison check lint precommit document install clean help

# Default target
help:
	@echo "embedmit Makefile targets:"
	@echo "  make test            - Run unit tests only"
	@echo "  make test-all        - Run lint + unit tests + comparison tests"
	@echo "  make test-comparison - Run comparison tests against embed package"
	@echo "  make check           - Full R CMD check"
	@echo "  make lint            - Run lintr for code style"
	@echo "  make precommit       - Pre-commit test suite (same as test-all)"
	@echo "  make document        - Update documentation with roxygen2"
	@echo "  make install         - Install package locally"
	@echo "  make clean           - Remove build artifacts"

# Run unit tests only
test:
	@echo "=========================================="
	@echo "Running unit tests..."
	@echo "=========================================="
	Rscript -e "devtools::test()"

# Run lint checks
lint:
	@echo "=========================================="
	@echo "Running lintr checks..."
	@echo "=========================================="
	Rscript -e "if (requireNamespace('lintr', quietly = TRUE)) lintr::lint_package() else message('lintr not installed, skipping')"

# Run comparison tests against embed package
test-comparison:
	@echo "=========================================="
	@echo "Running comparison tests vs embed..."
	@echo "=========================================="
	Rscript -e "testthat::test_file('tests/testthat/test-zzz_comparison_embed.R')"

# Run all tests (lint + unit + comparison)
test-all: lint test test-comparison
	@echo "=========================================="
	@echo "ALL TESTS PASSED"
	@echo "=========================================="

# Pre-commit test suite
precommit:
	@echo "=========================================="
	@echo "PRE-COMMIT TEST SUITE"
	@echo "=========================================="
	@./scripts/precommit-tests.sh

# Full R CMD check
check:
	@echo "=========================================="
	@echo "Running R CMD check..."
	@echo "=========================================="
	Rscript -e "devtools::check()"

# Update documentation
document:
	@echo "=========================================="
	@echo "Updating documentation..."
	@echo "=========================================="
	Rscript -e "devtools::document()"

# Install package locally
install:
	@echo "=========================================="
	@echo "Installing package..."
	@echo "=========================================="
	Rscript -e "devtools::install()"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf src/*.o src/*.so *.Rcheck man/*.Rd
