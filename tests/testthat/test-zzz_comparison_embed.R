# Comparison tests: embedmit vs embed
# These tests verify that embedmit produces equivalent or statistically similar
# results to the original embed package.

# =============================================================================
# Section 1: step_lencode_glm Exact Equivalence
# =============================================================================

test_that("step_lencode_glm produces exact same results as embed", {
  skip_if_no_embed()
  skip_on_cran()

  test_data <- create_recipe_test_data(n = 300, seed = 42)
  train_idx <- 1:200
  test_idx <- 201:300
  train_data <- test_data[train_idx, ]
  test_data_new <- test_data[test_idx, ]

  # embedmit recipe
  rec_embedmit <- recipes::recipe(outcome_num ~ ., data = train_data) |>
    embedmit::step_lencode_glm(cat_low, cat_med, outcome = vars(outcome_num)) |>
    recipes::prep(training = train_data)

  result_embedmit <- recipes::bake(rec_embedmit, new_data = test_data_new)

  # embed recipe
  rec_embed <- recipes::recipe(outcome_num ~ ., data = train_data) |>
    embed::step_lencode_glm(cat_low, cat_med, outcome = vars(outcome_num)) |>
    recipes::prep(training = train_data)

  result_embed <- recipes::bake(rec_embed, new_data = test_data_new)

  # Compare numeric results
  comparison <- compare_recipe_results_exact(result_embedmit, result_embed, tolerance = 1e-10)

  expect_true(
    comparison$equivalent,
    info = format_exact_comparison(comparison)
  )

  # Compare tidy output
  tidy_embedmit <- recipes::tidy(rec_embedmit, number = 1)
  tidy_embed <- recipes::tidy(rec_embed, number = 1)

  tidy_comparison <- compare_tidy_output(tidy_embedmit, tidy_embed, tolerance = 1e-10)
  expect_true(tidy_comparison$equivalent, info = tidy_comparison$reason)
})

test_that("step_lencode_glm with factor outcome matches embed", {
  skip_if_no_embed()
  skip_on_cran()

  test_data <- create_recipe_test_data(n = 300, seed = 43)
  train_data <- test_data[1:200, ]

  # embedmit
  rec_embedmit <- recipes::recipe(outcome_cat ~ ., data = train_data) |>
    embedmit::step_lencode_glm(cat_low, outcome = vars(outcome_cat)) |>
    recipes::prep(training = train_data)

  result_embedmit <- recipes::bake(rec_embedmit, new_data = NULL)

  # embed
  rec_embed <- recipes::recipe(outcome_cat ~ ., data = train_data) |>
    embed::step_lencode_glm(cat_low, outcome = vars(outcome_cat)) |>
    recipes::prep(training = train_data)

  result_embed <- recipes::bake(rec_embed, new_data = NULL)

  comparison <- compare_recipe_results_exact(result_embedmit, result_embed, tolerance = 1e-10)
  expect_true(comparison$equivalent, info = format_exact_comparison(comparison))
})

# =============================================================================
# Section 2: step_lencode_mixed Exact Equivalence
# =============================================================================

test_that("step_lencode_mixed produces exact same results as embed", {
  skip_if_no_embed()
  skip_on_cran()
  skip_if_not_installed("lme4")

  test_data <- create_recipe_test_data(n = 300, seed = 44)
  train_data <- test_data[1:200, ]
  test_data_new <- test_data[201:300, ]

  # embedmit
  rec_embedmit <- recipes::recipe(outcome_num ~ ., data = train_data) |>
    embedmit::step_lencode_mixed(cat_low, cat_med, outcome = vars(outcome_num)) |>
    recipes::prep(training = train_data)

  result_embedmit <- recipes::bake(rec_embedmit, new_data = test_data_new)

  # embed
  rec_embed <- recipes::recipe(outcome_num ~ ., data = train_data) |>
    embed::step_lencode_mixed(cat_low, cat_med, outcome = vars(outcome_num)) |>
    recipes::prep(training = train_data)

  result_embed <- recipes::bake(rec_embed, new_data = test_data_new)

  comparison <- compare_recipe_results_exact(result_embedmit, result_embed, tolerance = 1e-8)
  expect_true(comparison$equivalent, info = format_exact_comparison(comparison))
})

# =============================================================================
# Section 3: step_discretize_cart Exact Equivalence
# =============================================================================

test_that("step_discretize_cart produces exact same results as embed", {
  skip_if_no_embed()
  skip_on_cran()
  skip_if_not_installed("rpart")

  test_data <- create_recipe_test_data(n = 300, seed = 45)
  train_data <- test_data[1:200, ]
  test_data_new <- test_data[201:300, ]

  # embedmit
  rec_embedmit <- recipes::recipe(outcome_cat ~ ., data = train_data) |>
    embedmit::step_discretize_cart(x1, x2, outcome = vars(outcome_cat)) |>
    recipes::prep(training = train_data)

  result_embedmit <- recipes::bake(rec_embedmit, new_data = test_data_new)

  # embed
  rec_embed <- recipes::recipe(outcome_cat ~ ., data = train_data) |>
    embed::step_discretize_cart(x1, x2, outcome = vars(outcome_cat)) |>
    recipes::prep(training = train_data)

  result_embed <- recipes::bake(rec_embed, new_data = test_data_new)

  # Factor columns should match
  factor_comparison <- compare_factor_columns(result_embedmit, result_embed)
  expect_true(
    factor_comparison$equivalent,
    info = sprintf("Mismatched factor columns: %s", paste(factor_comparison$mismatched_cols, collapse = ", "))
  )
})

# =============================================================================
# Section 4: step_collapse_cart Exact Equivalence
# =============================================================================

test_that("step_collapse_cart produces exact same results as embed", {
  skip_if_no_embed()
  skip_on_cran()
  skip_if_not_installed("rpart")

  test_data <- create_recipe_test_data(n = 300, seed = 46)
  train_data <- test_data[1:200, ]
  test_data_new <- test_data[201:300, ]

  # embedmit
  rec_embedmit <- recipes::recipe(outcome_cat ~ ., data = train_data) |>
    embedmit::step_collapse_cart(cat_high, outcome = vars(outcome_cat)) |>
    recipes::prep(training = train_data)

  result_embedmit <- recipes::bake(rec_embedmit, new_data = test_data_new)

  # embed
  rec_embed <- recipes::recipe(outcome_cat ~ ., data = train_data) |>
    embed::step_collapse_cart(cat_high, outcome = vars(outcome_cat)) |>
    recipes::prep(training = train_data)

  result_embed <- recipes::bake(rec_embed, new_data = test_data_new)

  factor_comparison <- compare_factor_columns(result_embedmit, result_embed)
  expect_true(factor_comparison$equivalent, info = sprintf("Mismatched: %s", paste(factor_comparison$mismatched_cols, collapse = ", ")))
})

# =============================================================================
# Section 5: step_collapse_stringdist Exact Equivalence
# =============================================================================

test_that("step_collapse_stringdist produces exact same results as embed", {
  skip_if_no_embed()
  skip_on_cran()
  skip_if_not_installed("stringdist")

  test_data <- create_recipe_test_data(n = 300, seed = 47)
  train_data <- test_data[1:200, ]
  test_data_new <- test_data[201:300, ]

  # embedmit
  rec_embedmit <- recipes::recipe(outcome_cat ~ ., data = train_data) |>
    embedmit::step_collapse_stringdist(cat_high, distance = 3) |>
    recipes::prep(training = train_data)

  result_embedmit <- recipes::bake(rec_embedmit, new_data = test_data_new)

  # embed
  rec_embed <- recipes::recipe(outcome_cat ~ ., data = train_data) |>
    embed::step_collapse_stringdist(cat_high, distance = 3) |>
    recipes::prep(training = train_data)

  result_embed <- recipes::bake(rec_embed, new_data = test_data_new)

  factor_comparison <- compare_factor_columns(result_embedmit, result_embed)
  expect_true(factor_comparison$equivalent)
})

# =============================================================================
# Section 6: step_woe Exact Equivalence
# =============================================================================

test_that("step_woe produces exact same results as embed", {
  skip_if_no_embed()
  skip_on_cran()

  test_data <- create_binned_test_data(n = 500, seed = 48)
  train_data <- test_data[1:400, ]
  test_data_new <- test_data[401:500, ]

  # embedmit
  rec_embedmit <- recipes::recipe(outcome ~ ., data = train_data) |>
    embedmit::step_woe(cat1, cat2, outcome = vars(outcome)) |>
    recipes::prep(training = train_data)

  result_embedmit <- recipes::bake(rec_embedmit, new_data = test_data_new)

  # embed
  rec_embed <- recipes::recipe(outcome ~ ., data = train_data) |>
    embed::step_woe(cat1, cat2, outcome = vars(outcome)) |>
    recipes::prep(training = train_data)

  result_embed <- recipes::bake(rec_embed, new_data = test_data_new)

  comparison <- compare_recipe_results_exact(result_embedmit, result_embed, tolerance = 1e-10)
  expect_true(comparison$equivalent, info = format_exact_comparison(comparison))

  # Compare tidy output (dictionary)
  tidy_embedmit <- recipes::tidy(rec_embedmit, number = 1)
  tidy_embed <- recipes::tidy(rec_embed, number = 1)

  tidy_comparison <- compare_tidy_output(tidy_embedmit, tidy_embed, tolerance = 1e-10)
  expect_true(tidy_comparison$equivalent, info = tidy_comparison$reason)
})

# =============================================================================
# Section 7: step_pca_truncated Exact Equivalence
# =============================================================================

test_that("step_pca_truncated produces exact same results as embed", {
  skip_if_no_embed()
  skip_on_cran()
  skip_if_not_installed("irlba")

  test_data <- create_recipe_test_data(n = 200, seed = 49)
  train_data <- test_data[1:150, ]
  test_data_new <- test_data[151:200, ]

  # embedmit
  set.seed(100)
  rec_embedmit <- recipes::recipe(outcome_cat ~ ., data = train_data) |>
    embedmit::step_pca_truncated(x1, x2, x3, x4, num_comp = 2) |>
    recipes::prep(training = train_data)

  result_embedmit <- recipes::bake(rec_embedmit, new_data = test_data_new)

  # embed
  set.seed(100)
  rec_embed <- recipes::recipe(outcome_cat ~ ., data = train_data) |>
    embed::step_pca_truncated(x1, x2, x3, x4, num_comp = 2) |>
    recipes::prep(training = train_data)

  result_embed <- recipes::bake(rec_embed, new_data = test_data_new)

  # PCA results may differ in sign but should have same absolute values
  pca_cols <- grep("^PC", names(result_embedmit), value = TRUE)
  for (col in pca_cols) {
    # Check if values match or are negated (sign ambiguity)
    same_sign <- max(abs(result_embedmit[[col]] - result_embed[[col]])) < 1e-8
    diff_sign <- max(abs(result_embedmit[[col]] + result_embed[[col]])) < 1e-8

    expect_true(
      same_sign || diff_sign,
      info = sprintf("Column %s differs beyond sign ambiguity", col)
    )
  }
})

# =============================================================================
# Section 8: step_pca_sparse Exact Equivalence
# =============================================================================

test_that("step_pca_sparse produces exact same results as embed", {
  skip_if_no_embed()
  skip_on_cran()
  skip_if_not_installed("irlba")

  test_data <- create_recipe_test_data(n = 200, seed = 50)
  train_data <- test_data[1:150, ]
  test_data_new <- test_data[151:200, ]

  # embedmit
  set.seed(101)
  rec_embedmit <- recipes::recipe(outcome_cat ~ ., data = train_data) |>
    embedmit::step_pca_sparse(x1, x2, x3, x4, num_comp = 2, predictor_prop = 0.5) |>
    recipes::prep(training = train_data)

  result_embedmit <- recipes::bake(rec_embedmit, new_data = test_data_new)

  # embed
  set.seed(101)
  rec_embed <- recipes::recipe(outcome_cat ~ ., data = train_data) |>
    embed::step_pca_sparse(x1, x2, x3, x4, num_comp = 2, predictor_prop = 0.5) |>
    recipes::prep(training = train_data)

  result_embed <- recipes::bake(rec_embed, new_data = test_data_new)

  pca_cols <- grep("^PC", names(result_embedmit), value = TRUE)
  for (col in pca_cols) {
    same_sign <- max(abs(result_embedmit[[col]] - result_embed[[col]])) < 1e-8
    diff_sign <- max(abs(result_embedmit[[col]] + result_embed[[col]])) < 1e-8
    expect_true(same_sign || diff_sign, info = sprintf("Column %s differs", col))
  }
})

# =============================================================================
# Section 9: step_umap Statistical Similarity
# =============================================================================

test_that("step_umap produces statistically similar results to embed", {
  skip_if_no_embed()
  skip_if_no_uwot()
  skip_on_cran()
  skip_if_not_installed("irlba")

  test_data <- create_umap_test_data(n = 120, p = 4, seed = 51)
  train_data <- test_data[1:100, ]
  test_data_new <- test_data[101:120, ]

  # Get numeric columns for UMAP comparison
  numeric_cols <- c("x1", "x2", "x3", "x4")

  # embedmit
  set.seed(102)
  rec_embedmit <- recipes::recipe(outcome ~ ., data = train_data) |>
    embedmit::step_umap(
      recipes::all_numeric_predictors(),
      num_comp = 2,
      neighbors = 10,
      min_dist = 0.1,
      epochs = 50
    ) |>
    recipes::prep(training = train_data)

  result_embedmit <- recipes::bake(rec_embedmit, new_data = NULL)

  # embed
  set.seed(102)
  rec_embed <- recipes::recipe(outcome ~ ., data = train_data) |>
    embed::step_umap(
      recipes::all_numeric_predictors(),
      num_comp = 2,
      neighbors = 10,
      min_dist = 0.1,
      epochs = 50
    ) |>
    recipes::prep(training = train_data)

  result_embed <- recipes::bake(rec_embed, new_data = NULL)

  # Extract UMAP columns
  umap_cols_embedmit <- grep("^UMAP|^umap", names(result_embedmit), value = TRUE)
  umap_cols_embed <- grep("^UMAP|^umap", names(result_embed), value = TRUE)

  expect_equal(length(umap_cols_embedmit), 2)
  expect_equal(length(umap_cols_embed), 2)

  # Compare embeddings using quality metrics
  embed_embedmit <- as.matrix(result_embedmit[, umap_cols_embedmit])
  embed_embed <- as.matrix(result_embed[, umap_cols_embed])
  original_data <- as.matrix(train_data[, numeric_cols])

  comparison <- compare_umap_embeddings(embed_embedmit, embed_embed, original_data, k = 8)

  message(format_umap_comparison(comparison))

  # Both should have reasonable trustworthiness
  expect_true(
    comparison$trust_embedmit >= 0.70,
    info = sprintf("embedmit trustworthiness too low: %.4f", comparison$trust_embedmit)
  )
  expect_true(
    comparison$trust_embed >= 0.70,
    info = sprintf("embed trustworthiness too low: %.4f", comparison$trust_embed)
  )

  # Trustworthiness difference should be small
  expect_true(
    comparison$trust_diff < 0.2,
    info = sprintf("Trustworthiness difference too large: %.4f", comparison$trust_diff)
  )
})

test_that("step_umap with supervised outcome matches embed quality", {
  skip_if_no_embed()
  skip_if_no_uwot()
  skip_on_cran()
  skip_if_not_installed("irlba")

  test_data <- create_umap_test_data(n = 120, p = 4, seed = 52)
  train_data <- test_data[1:100, ]

  numeric_cols <- c("x1", "x2", "x3", "x4")

  # embedmit with outcome
  set.seed(103)
  rec_embedmit <- recipes::recipe(outcome ~ ., data = train_data) |>
    embedmit::step_umap(
      recipes::all_numeric_predictors(),
      outcome = vars(outcome),
      num_comp = 2,
      neighbors = 10,
      min_dist = 0.1,
      epochs = 50
    ) |>
    recipes::prep(training = train_data)

  result_embedmit <- recipes::bake(rec_embedmit, new_data = NULL)

  # embed with outcome
  set.seed(103)
  rec_embed <- recipes::recipe(outcome ~ ., data = train_data) |>
    embed::step_umap(
      recipes::all_numeric_predictors(),
      outcome = vars(outcome),
      num_comp = 2,
      neighbors = 10,
      min_dist = 0.1,
      epochs = 50
    ) |>
    recipes::prep(training = train_data)

  result_embed <- recipes::bake(rec_embed, new_data = NULL)

  umap_cols_embedmit <- grep("^UMAP|^umap", names(result_embedmit), value = TRUE)
  umap_cols_embed <- grep("^UMAP|^umap", names(result_embed), value = TRUE)

  embed_embedmit <- as.matrix(result_embedmit[, umap_cols_embedmit])
  embed_embed <- as.matrix(result_embed[, umap_cols_embed])
  original_data <- as.matrix(train_data[, numeric_cols])

  comparison <- compare_umap_embeddings(embed_embedmit, embed_embed, original_data, k = 8)

  expect_true(comparison$trust_embedmit >= 0.65)
  expect_true(comparison$trust_embed >= 0.65)
  expect_true(comparison$trust_diff < 0.25)
})

# =============================================================================
# Section 10: step_discretize_xgb Exact Equivalence
# =============================================================================

test_that("step_discretize_xgb produces exact same results as embed", {
  skip_if_no_embed()
  skip_on_cran()
  skip_if_not_installed("xgboost")

  test_data <- create_recipe_test_data(n = 300, seed = 53)
  train_data <- test_data[1:200, ]
  test_data_new <- test_data[201:300, ]

  # embedmit
  set.seed(104)
  rec_embedmit <- recipes::recipe(outcome_cat ~ ., data = train_data) |>
    embedmit::step_discretize_xgb(x1, x2, outcome = vars(outcome_cat)) |>
    recipes::prep(training = train_data)

  result_embedmit <- recipes::bake(rec_embedmit, new_data = test_data_new)

  # embed
  set.seed(104)
  rec_embed <- recipes::recipe(outcome_cat ~ ., data = train_data) |>
    embed::step_discretize_xgb(x1, x2, outcome = vars(outcome_cat)) |>
    recipes::prep(training = train_data)

  result_embed <- recipes::bake(rec_embed, new_data = test_data_new)

  factor_comparison <- compare_factor_columns(result_embedmit, result_embed)
  expect_true(factor_comparison$equivalent)
})

# =============================================================================
# Section 11: Performance Comparison
# =============================================================================

test_that("embedmit performance is comparable to embed", {
  skip_if_no_embed()
  skip_on_cran()
  skip("Performance test - run manually")

  test_data <- create_recipe_test_data(n = 1000, seed = 54)
  train_data <- test_data

  # Time embedmit step_lencode_glm
  time_embedmit <- system.time({
    rec <- recipes::recipe(outcome_num ~ ., data = train_data) |>
      embedmit::step_lencode_glm(cat_low, cat_med, cat_high, outcome = vars(outcome_num)) |>
      recipes::prep(training = train_data)
    recipes::bake(rec, new_data = NULL)
  })["elapsed"]

  # Time embed step_lencode_glm
  time_embed <- system.time({
    rec <- recipes::recipe(outcome_num ~ ., data = train_data) |>
      embed::step_lencode_glm(cat_low, cat_med, cat_high, outcome = vars(outcome_num)) |>
      recipes::prep(training = train_data)
    recipes::bake(rec, new_data = NULL)
  })["elapsed"]

  message(sprintf("embedmit: %.3fs, embed: %.3fs, ratio: %.2f",
                  time_embedmit, time_embed, time_embedmit / time_embed))

  # embedmit should be within 2x of embed
  expect_true(
    time_embedmit < time_embed * 2,
    info = sprintf("embedmit (%.3fs) is more than 2x slower than embed (%.3fs)",
                   time_embedmit, time_embed)
  )
})

# =============================================================================
# Section 12: Edge Cases
# =============================================================================

test_that("embedmit handles empty factor levels like embed", {
  skip_if_no_embed()
  skip_on_cran()

  test_data <- create_recipe_test_data(n = 100, seed = 55)
  # Create empty levels
  test_data$cat_low <- factor(test_data$cat_low, levels = c(levels(test_data$cat_low), "D", "E"))
  train_data <- test_data

  # embedmit
  rec_embedmit <- recipes::recipe(outcome_num ~ ., data = train_data) |>
    embedmit::step_lencode_glm(cat_low, outcome = vars(outcome_num)) |>
    recipes::prep(training = train_data)

  result_embedmit <- recipes::bake(rec_embedmit, new_data = NULL)

  # embed
  rec_embed <- recipes::recipe(outcome_num ~ ., data = train_data) |>
    embed::step_lencode_glm(cat_low, outcome = vars(outcome_num)) |>
    recipes::prep(training = train_data)

  result_embed <- recipes::bake(rec_embed, new_data = NULL)

  comparison <- compare_recipe_results_exact(result_embedmit, result_embed, tolerance = 1e-10)
  expect_true(comparison$equivalent)
})

test_that("embedmit handles NA values like embed", {
  skip_if_no_embed()
  skip_on_cran()

  test_data <- create_recipe_test_data(n = 100, seed = 56)
  # Introduce NAs
  test_data$x1[sample(100, 10)] <- NA
  test_data$cat_low[sample(100, 5)] <- NA
  train_data <- test_data

  # Both should handle NAs similarly (may warn or handle gracefully)
  expect_no_error({
    rec_embedmit <- recipes::recipe(outcome_num ~ ., data = train_data) |>
      embedmit::step_lencode_glm(cat_low, outcome = vars(outcome_num)) |>
      recipes::prep(training = train_data)
  })

  expect_no_error({
    rec_embed <- recipes::recipe(outcome_num ~ ., data = train_data) |>
      embed::step_lencode_glm(cat_low, outcome = vars(outcome_num)) |>
      recipes::prep(training = train_data)
  })
})

test_that("embedmit handles single-level factors like embed", {
  skip_if_no_embed()
  skip_on_cran()

  test_data <- create_recipe_test_data(n = 100, seed = 57)
  # Create single-level factor
  test_data$single_level <- factor(rep("only_level", 100))
  train_data <- test_data

  # Both should handle this edge case similarly
  # Either both should work or both should error/warn
  embedmit_error <- NULL
  embed_error <- NULL

  tryCatch({
    rec_embedmit <- recipes::recipe(outcome_num ~ ., data = train_data) |>
      embedmit::step_lencode_glm(single_level, outcome = vars(outcome_num)) |>
      recipes::prep(training = train_data)
  }, error = function(e) {
    embedmit_error <<- conditionMessage(e)
  })

  tryCatch({
    rec_embed <- recipes::recipe(outcome_num ~ ., data = train_data) |>
      embed::step_lencode_glm(single_level, outcome = vars(outcome_num)) |>
      recipes::prep(training = train_data)
  }, error = function(e) {
    embed_error <<- conditionMessage(e)
  })

  # Both should behave similarly (both error or both succeed)
  expect_equal(
    is.null(embedmit_error),
    is.null(embed_error),
    info = sprintf("Different error behavior: embedmit=%s, embed=%s",
                   ifelse(is.null(embedmit_error), "success", embedmit_error),
                   ifelse(is.null(embed_error), "success", embed_error))
  )
})

# =============================================================================
# Section 13: Tidy Method Consistency
# =============================================================================

test_that("tidy methods return equivalent structures", {
  skip_if_no_embed()
  skip_on_cran()

  test_data <- create_recipe_test_data(n = 200, seed = 58)
  train_data <- test_data[1:150, ]

  # embedmit
  rec_embedmit <- recipes::recipe(outcome_num ~ ., data = train_data) |>
    embedmit::step_lencode_glm(cat_low, cat_med, outcome = vars(outcome_num)) |>
    recipes::prep(training = train_data)

  # embed
  rec_embed <- recipes::recipe(outcome_num ~ ., data = train_data) |>
    embed::step_lencode_glm(cat_low, cat_med, outcome = vars(outcome_num)) |>
    recipes::prep(training = train_data)

  # Compare tidy output structure
  tidy_embedmit <- recipes::tidy(rec_embedmit, number = 1)
  tidy_embed <- recipes::tidy(rec_embed, number = 1)

  # Same columns
  expect_equal(names(tidy_embedmit), names(tidy_embed))

  # Same number of rows
  expect_equal(nrow(tidy_embedmit), nrow(tidy_embed))

  # Same data types
  for (col in names(tidy_embedmit)) {
    expect_equal(
      class(tidy_embedmit[[col]]),
      class(tidy_embed[[col]]),
      info = sprintf("Different class for column %s", col)
    )
  }
})

# =============================================================================
# Section 14: Required Packages Consistency
# =============================================================================

test_that("required_pkgs returns same packages as embed", {
  skip_if_no_embed()
  skip_on_cran()

  test_data <- create_recipe_test_data(n = 50, seed = 59)

  # step_lencode_glm
  rec_embedmit <- recipes::recipe(outcome_num ~ ., data = test_data) |>
    embedmit::step_lencode_glm(cat_low, outcome = vars(outcome_num))

  rec_embed <- recipes::recipe(outcome_num ~ ., data = test_data) |>
    embed::step_lencode_glm(cat_low, outcome = vars(outcome_num))

  pkgs_embedmit <- generics::required_pkgs(rec_embedmit)
  pkgs_embed <- generics::required_pkgs(rec_embed)

  # Exclude both package names from comparison
  # (embedmit may transitively depend on embed)
  pkgs_embedmit_core <- setdiff(pkgs_embedmit, c("embedmit", "embed"))
  pkgs_embed_core <- setdiff(pkgs_embed, c("embedmit", "embed"))

  expect_equal(
    sort(pkgs_embedmit_core),
    sort(pkgs_embed_core),
    info = "Different core required packages (excluding embedmit/embed)"
  )
})
