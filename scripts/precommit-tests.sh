#!/bin/bash
# Pre-commit test script for embedmit
# Runs all tests and blocks commit if any fail

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo ""
echo "=========================================="
echo "  embedmit PRE-COMMIT TEST SUITE"
echo "=========================================="
echo ""

# Track overall status
FAILED=0

# Step 1: Lint check (optional - don't fail if lintr not installed)
echo -e "${YELLOW}Step 1/3: Running lint checks...${NC}"
if Rscript -e "if (requireNamespace('lintr', quietly = TRUE)) { lints <- lintr::lint_package(); if (length(lints) > 0) { print(lints); quit(status = 1) } } else { message('lintr not installed, skipping') }" 2>&1; then
    echo -e "${GREEN}✓ Lint checks passed${NC}"
else
    echo -e "${YELLOW}⚠ Lint warnings (non-blocking)${NC}"
fi
echo ""

# Step 2: Unit tests
echo -e "${YELLOW}Step 2/3: Running unit tests...${NC}"
if Rscript -e "
  library(testthat)
  library(devtools)
  results <- devtools::test(reporter = 'summary')
  df <- as.data.frame(results)
  failed <- sum(df\$failed)
  errors <- sum(df\$error)
  if (failed > 0 || errors > 0) {
    cat('\nFailed:', failed, 'Errors:', errors, '\n')
    quit(status = 1)
  }
" 2>&1; then
    echo -e "${GREEN}✓ Unit tests passed${NC}"
else
    echo -e "${RED}✗ Unit tests FAILED${NC}"
    FAILED=1
fi
echo ""

# Step 3: Comparison tests (skip if embed not installed)
echo -e "${YELLOW}Step 3/3: Running comparison tests...${NC}"
if Rscript -e "
  if (!requireNamespace('embed', quietly = TRUE)) {
    message('embed package not installed, skipping comparison tests')
    quit(status = 0)
  }
  library(testthat)
  results <- testthat::test_file('tests/testthat/test-zzz_comparison_embed.R', reporter = 'summary')
  df <- as.data.frame(results)
  failed <- sum(df\$failed)
  errors <- sum(df\$error)
  if (failed > 0 || errors > 0) {
    cat('\nFailed:', failed, 'Errors:', errors, '\n')
    quit(status = 1)
  }
" 2>&1; then
    echo -e "${GREEN}✓ Comparison tests passed${NC}"
else
    echo -e "${RED}✗ Comparison tests FAILED${NC}"
    FAILED=1
fi
echo ""

# Final result
echo "=========================================="
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}  ALL TESTS PASSED - OK TO COMMIT${NC}"
    echo "=========================================="
    exit 0
else
    echo -e "${RED}  TESTS FAILED - COMMIT BLOCKED${NC}"
    echo "=========================================="
    echo ""
    echo "Fix the failing tests before committing."
    echo "To bypass (emergency only): git commit --no-verify"
    exit 1
fi
