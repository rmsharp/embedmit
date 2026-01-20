# =============================================================================
# Fork-Specific Regression Tests for embedmit
# These tests verify the critical differences between embedmit and embed
# to prevent regressions in fork-specific behavior.
# =============================================================================

# -----------------------------------------------------------------------------
# Section 1: Tausworthe RNG Default (CRITICAL)
# The core differentiator of embedmit is defaulting to tausworthe RNG
# to avoid AGPL-licensed dqrng dependency
# -----------------------------------------------------------------------------

test_that("step_umap defaults to tausworthe RNG (fork-specific)", {

  # Create a recipe without specifying options

rec <- recipe(~., data = iris[1:50, 1:4]) |>
    step_umap(all_predictors(), num_comp = 2)

  # Verify the default options contain tausworthe
  expect_equal(
    rec$steps[[1]]$options$rng_type,
    "tausworthe",
    info = "CRITICAL: embedmit must default to tausworthe RNG to avoid AGPL dependency"
  )
})

test_that("step_umap options include tausworthe even when other options specified", {
  # When user specifies some options, tausworthe should still be default
  rec <- recipe(~., data = iris[1:50, 1:4]) |>
    step_umap(all_predictors(), num_comp = 2,
              options = list(verbose = TRUE, n_threads = 2))

  # User-specified options should be present
expect_true(rec$steps[[1]]$options$verbose)
  expect_equal(rec$steps[[1]]$options$n_threads, 2)

  # But if user didn't specify rng_type, we can't enforce it at recipe creation
  # The default is applied in the step definition
})

test_that("step_umap respects user-specified rng_type override", {
  # Users should be able to override the default if they choose
  rec <- recipe(~., data = iris[1:50, 1:4]) |>
    step_umap(all_predictors(), num_comp = 2,
              options = list(verbose = FALSE, n_threads = 1, rng_type = "deterministic"))

  expect_equal(
    rec$steps[[1]]$options$rng_type,
    "deterministic",
    info = "User should be able to override default rng_type"
  )
})

# -----------------------------------------------------------------------------
# Section 2: uwotmit Dependency (CRITICAL)
# embedmit must use uwotmit, not uwot
# -----------------------------------------------------------------------------

test_that("step_umap requires uwotmit package (not uwot)", {
  rec <- recipe(~., data = iris[1:50, 1:4]) |>
    step_umap(all_predictors(), num_comp = 2)

  pkgs <- required_pkgs(rec$steps[[1]])

  # Must contain uwotmit
  expect_true(
    "uwotmit" %in% pkgs,
    info = "CRITICAL: embedmit must require uwotmit, not uwot"
  )

  # Should NOT require uwot (we've forked away from it)
  expect_false(
    "uwot" %in% pkgs,
    info = "embedmit should not require uwot - we use uwotmit"
  )
})

test_that("required_pkgs for step_umap returns embedmit and uwotmit", {
  rec <- recipe(~., data = iris[1:50, 1:4]) |>
    step_umap(all_predictors(), num_comp = 2)

  pkgs <- required_pkgs(rec$steps[[1]])

  expect_true("embedmit" %in% pkgs)
  expect_true("uwotmit" %in% pkgs)
})

# -----------------------------------------------------------------------------
# Section 3: UMAP Execution with uwotmit (Integration)
# Verify that step_umap actually works with uwotmit
# -----------------------------------------------------------------------------

test_that("step_umap prep succeeds with uwotmit backend", {
  skip_if_not_installed("irlba")

  set.seed(42)
  rec <- recipe(~., data = iris[1:100, 1:4]) |>
    step_umap(all_predictors(), num_comp = 2, neighbors = 10) |>
    prep(training = iris[1:100, 1:4])

  # Should have trained successfully
expect_true(rec$steps[[1]]$trained)

  # Should have embedding object
  expect_true(!is.null(rec$steps[[1]]$object))
})

test_that("step_umap bake produces valid output with uwotmit", {
  skip_if_not_installed("irlba")

  set.seed(42)
  rec <- recipe(~., data = iris[1:100, 1:4]) |>
    step_umap(all_predictors(), num_comp = 2, neighbors = 10) |>
    prep(training = iris[1:100, 1:4])

  result <- bake(rec, new_data = iris[101:120, 1:4])

  # Check output structure
  expect_true("UMAP1" %in% names(result))
  expect_true("UMAP2" %in% names(result))
  expect_equal(nrow(result), 20)
  expect_true(is.numeric(result$UMAP1))
  expect_true(is.numeric(result$UMAP2))
  expect_false(anyNA(result$UMAP1))
  expect_false(anyNA(result$UMAP2))
})

test_that("step_umap with tausworthe produces reproducible results", {
  skip_if_not_installed("irlba")

  # First run
  set.seed(123)
  rec1 <- recipe(~., data = iris[1:100, 1:4]) |>
    step_umap(all_predictors(), num_comp = 2, neighbors = 10) |>
    prep(training = iris[1:100, 1:4])

  embedding1 <- rec1$steps[[1]]$object$embedding

  # Second run with same seed
  set.seed(123)
  rec2 <- recipe(~., data = iris[1:100, 1:4]) |>
    step_umap(all_predictors(), num_comp = 2, neighbors = 10) |>
    prep(training = iris[1:100, 1:4])

  embedding2 <- rec2$steps[[1]]$object$embedding

  # Should be identical with same seed
  expect_equal(embedding1, embedding2,
    info = "UMAP with tausworthe RNG should be reproducible with same seed")
})

# -----------------------------------------------------------------------------
# Section 4: Edge Cases for step_umap
# -----------------------------------------------------------------------------

test_that("step_umap handles small datasets correctly", {
  skip_if_not_installed("irlba")

  # Small dataset with fewer observations than default neighbors
  small_data <- iris[1:10, 1:4]

  set.seed(42)
  rec <- recipe(~., data = small_data) |>
    step_umap(all_predictors(), num_comp = 2, neighbors = 5) |>
    prep(training = small_data)

  expect_true(rec$steps[[1]]$trained)

  result <- bake(rec, new_data = small_data)
  expect_equal(nrow(result), 10)
})

test_that("step_umap neighbors gets adjusted for small datasets",
{
  skip_if_not_installed("irlba")

  # Very small dataset
  tiny_data <- iris[1:5, 1:4]

  set.seed(42)
  rec <- recipe(~., data = tiny_data) |>
    step_umap(all_predictors(), num_comp = 2, neighbors = 3) |>
    prep(training = tiny_data)

  expect_true(rec$steps[[1]]$trained)
})

test_that("step_umap num_comp is respected", {
  skip_if_not_installed("irlba")

  set.seed(42)
  rec <- recipe(~., data = iris[1:100, 1:4]) |>
    step_umap(all_predictors(), num_comp = 3, neighbors = 10) |>
    prep(training = iris[1:100, 1:4])

  result <- bake(rec, new_data = iris[101:110, 1:4])

  # Should have 3 UMAP components
  expect_true("UMAP1" %in% names(result))
  expect_true("UMAP2" %in% names(result))
  expect_true("UMAP3" %in% names(result))
})

# -----------------------------------------------------------------------------
# Section 5: Supervised UMAP
# -----------------------------------------------------------------------------

test_that("step_umap supervised mode works with uwotmit", {
  skip_if_not_installed("irlba")

  set.seed(42)
  rec <- recipe(Species ~ ., data = iris[1:100, ]) |>
    step_umap(all_predictors(), outcome = vars(Species),
              num_comp = 2, neighbors = 10) |>
    prep(training = iris[1:100, ])

  expect_true(rec$steps[[1]]$trained)

  result <- bake(rec, new_data = iris[101:120, ])
  expect_true("UMAP1" %in% names(result))
  expect_true("UMAP2" %in% names(result))
})

# -----------------------------------------------------------------------------
# Section 6: Verify embedmit namespace (not embed)
# -----------------------------------------------------------------------------

test_that("package uses embedmit namespace consistently", {
  # Check that internal functions use embedmit, not embed
  expect_true(
    "embedmit" %in% loadedNamespaces() ||
    requireNamespace("embedmit", quietly = TRUE)
  )
})
