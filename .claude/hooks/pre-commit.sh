#!/bin/bash
# Pre-commit hook for embedmit - ensures TDD compliance

set -e

echo "Running embedmit pre-commit checks..."

# Change to project directory
cd "$(dirname "$0")/../.."

# Run R tests
echo "Running tests with devtools::test()..."
Rscript -e "
  library(testthat)
  library(devtools)

  # Run tests
  results <- devtools::test(reporter = 'summary')

  # Check for failures
  failed <- sum(as.data.frame(results)\$failed)
  errors <- sum(as.data.frame(results)\$error)

  if (failed > 0 || errors > 0) {
    cat('\n\nPRE-COMMIT FAILED: Tests have failures or errors\n')
    cat('Failed:', failed, 'Errors:', errors, '\n')
    quit(status = 1)
  }

  cat('\nAll tests passed!\n')
"

echo "Pre-commit checks passed!"
