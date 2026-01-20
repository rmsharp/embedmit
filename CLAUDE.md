# embedmit - Claude Code Project Configuration

## Project Overview

**embedmit** is an MIT-licensed fork of the `embed` package, providing
recipe steps for embedding categorical predictors and dimensionality
reduction in the tidymodels ecosystem.

------------------------------------------------------------------------

## MANDATORY: Claude Code Testing Requirements

**THIS SECTION IS NON-NEGOTIABLE. Claude MUST follow these rules.**

### Pre-Commit Testing is REQUIRED

Before ANY commit, Claude MUST run the full test suite:

``` bash
make test-all
```

If tests fail, Claude MUST fix them before committing. **NO
EXCEPTIONS.**

### Test Coverage Requirements by Feature Type

| Feature Type       | Required Tests                                            |
|--------------------|-----------------------------------------------------------|
| New recipe step    | Unit tests, prep/bake tests, tidy method test, edge cases |
| Bug fix            | Regression test that would have caught the bug            |
| API change         | Update existing tests, backward compatibility tests       |
| Performance change | Benchmark comparison, behavior equivalence tests          |

### Bug Fixes MUST Include Regression Tests

Every bug fix MUST include a test that: 1. Would have failed before the
fix 2. Passes after the fix 3. Prevents the bug from recurring

### How It Works

**When Claude is asked to commit:** 1. Claude MUST run `make test-all`
first 2. If tests fail, Claude fixes them before committing 3. The git
hook provides a safety net if Claude forgets

**When developing new features:** 1. Claude creates tests FIRST (TDD) 2.
Claude implements the feature 3. Claude verifies all tests pass before
marking work complete

**To manually verify tests:**

``` bash
make test-all      # Full test suite (lint + unit + comparison)
make precommit     # Same as above with detailed output
```

------------------------------------------------------------------------

## Development Standards

### Testing Commands

``` bash
# Makefile targets (PREFERRED)
make test          # Run unit tests only
make test-all      # Run lint + unit tests + comparison tests
make check         # Full R CMD check
make precommit     # Pre-commit test suite with clear output

# R commands (alternative)
Rscript -e "devtools::test()"
Rscript -e "devtools::check()"
Rscript -e "testthat::test_file('tests/testthat/test-zzz_comparison_embed.R')"
```

### Pre-Commit Checklist

Before committing code changes:

Run `make test-all` - all tests must pass

Run `make check` - no errors or warnings

New features have corresponding tests

Bug fixes include regression tests

Comparison tests pass (when embed package available)

### Code Style

- Follow tidyverse style guide
- Use roxygen2 for documentation
- Export S3 methods properly for recipes integration
- Maintain API compatibility with original embed package

### Key Files

- `R/` - Source code for recipe steps
- `tests/testthat/` - Test files
- `tests/testthat/helper_comparison.R` - Utilities for embed comparison
- `tests/testthat/test-zzz_comparison_embed.R` - Comparison test suite
- `Makefile` - Build and test automation
- `.githooks/pre-commit` - Git pre-commit hook

### Comparison Test Standards

When embed package is available, comparison tests verify: -
Deterministic functions produce exact matches (tolerance 1e-10) - UMAP
quality metrics: trustworthiness difference \< 0.1 - Both
implementations achieve \> 0.85 trustworthiness - Performance within 2x
of original

------------------------------------------------------------------------

## Git Hook Setup

To enable automatic pre-commit testing:

``` bash
git config core.hooksPath .githooks
```

This ensures `make precommit` runs before every commit. Can be bypassed
with `git commit --no-verify` in emergencies only.
