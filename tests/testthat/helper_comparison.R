# Comparison utility functions for embedmit vs embed testing

# Skip helper for embed comparison tests
skip_if_no_embed <- function() {
  testthat::skip_if_not(
    requireNamespace("embed", quietly = TRUE),
    "Package 'embed' not installed"
  )
}

# Skip helper for uwot comparison tests (for step_umap)
skip_if_no_uwot <- function() {
  testthat::skip_if_not(
    requireNamespace("uwot", quietly = TRUE),
    "Package 'uwot' not installed"
  )
}

# Compare two recipe results for exact equivalence
# Used for deterministic functions like step_lencode_glm
compare_recipe_results_exact <- function(result1, result2, tolerance = 1e-10) {
  # Get numeric columns
  numeric_cols <- sapply(result1, is.numeric)

  if (!any(numeric_cols)) {
    return(list(
      equivalent = identical(result1, result2),
      max_diff = NA,
      diff_cols = character(0)
    ))
  }

  max_diffs <- numeric(sum(numeric_cols))
  names(max_diffs) <- names(result1)[numeric_cols]

  for (col in names(result1)[numeric_cols]) {
    if (col %in% names(result2)) {
      max_diffs[col] <- max(abs(result1[[col]] - result2[[col]]), na.rm = TRUE)
    } else {
      max_diffs[col] <- Inf
    }
  }

  diff_cols <- names(max_diffs)[max_diffs > tolerance]

  list(
    equivalent = all(max_diffs <= tolerance, na.rm = TRUE),
    max_diff = max(max_diffs, na.rm = TRUE),
    diff_cols = diff_cols,
    col_diffs = max_diffs
  )
}

# Compare factor columns between two data frames
compare_factor_columns <- function(result1, result2) {
  factor_cols1 <- sapply(result1, is.factor)
  factor_cols2 <- sapply(result2, is.factor)

  if (sum(factor_cols1) != sum(factor_cols2)) {
    return(list(equivalent = FALSE, reason = "Different number of factor columns"))
  }

  factor_names <- names(result1)[factor_cols1]
  mismatches <- character(0)

  for (col in factor_names) {
    if (col %in% names(result2)) {
      if (!identical(as.character(result1[[col]]), as.character(result2[[col]]))) {
        mismatches <- c(mismatches, col)
      }
    } else {
      mismatches <- c(mismatches, col)
    }
  }

  list(
    equivalent = length(mismatches) == 0,
    mismatched_cols = mismatches
  )
}

# Compare tidy output from recipe steps
compare_tidy_output <- function(tidy1, tidy2, tolerance = 1e-10,
                                skip_cols = c("id")) {
  # Check same structure (excluding skipped columns)
  cols1 <- setdiff(names(tidy1), skip_cols)
  cols2 <- setdiff(names(tidy2), skip_cols)

  if (!identical(sort(cols1), sort(cols2))) {
    return(list(equivalent = FALSE, reason = "Different column names"))
  }

  if (nrow(tidy1) != nrow(tidy2)) {
    return(list(equivalent = FALSE, reason = "Different number of rows"))
  }

  # Compare each column (excluding skipped columns like 'id')
  for (col in cols1) {
    if (is.numeric(tidy1[[col]]) && is.numeric(tidy2[[col]])) {
      max_diff <- max(abs(tidy1[[col]] - tidy2[[col]]), na.rm = TRUE)
      if (max_diff > tolerance) {
        return(list(
          equivalent = FALSE,
          reason = sprintf("Column '%s' differs by %.2e", col, max_diff)
        ))
      }
    } else if (!identical(tidy1[[col]], tidy2[[col]])) {
      # For non-numeric, check exact equality
      if (!all(as.character(tidy1[[col]]) == as.character(tidy2[[col]]), na.rm = TRUE)) {
        return(list(equivalent = FALSE, reason = sprintf("Column '%s' values differ", col)))
      }
    }
  }

  list(equivalent = TRUE, reason = "All columns match")
}

# Compute k-nearest neighbor indices for a data matrix
compute_knn_indices <- function(X, k = 10) {
  n <- nrow(X)
  D <- as.matrix(stats::dist(X))
  t(apply(D, 1, function(row) order(row)[2:(k + 1)]))
}

# Trustworthiness metric for UMAP comparisons
trustworthiness <- function(X_high, X_low, k = 10) {
  n <- nrow(X_high)
  if (k >= n) k <- n - 1

  nn_high <- compute_knn_indices(X_high, k)
  nn_low <- compute_knn_indices(X_low, k)

  penalty <- 0
  for (i in seq_len(n)) {
    high_neighbors <- nn_high[i, ]
    low_neighbors <- nn_low[i, ]
    false_neighbors <- setdiff(low_neighbors, high_neighbors)

    if (length(false_neighbors) > 0) {
      D_high <- as.matrix(stats::dist(X_high))
      ranks <- rank(D_high[i, ])
      penalty <- penalty + sum(pmax(0, ranks[false_neighbors] - k))
    }
  }

  normalization <- n * k * (2 * n - 3 * k - 1)
  if (normalization <= 0) return(1)

  1 - (2 / normalization) * penalty
}

# Distance correlation between two embeddings
distance_correlation <- function(embed1, embed2) {
  D1 <- as.vector(stats::dist(embed1))
  D2 <- as.vector(stats::dist(embed2))

  if (length(D1) < 2) return(1)

  stats::cor(D1, D2, method = "spearman")
}

# Compare two UMAP embeddings for statistical similarity
compare_umap_embeddings <- function(embed1, embed2, original_data, k = 10) {
  trust1 <- trustworthiness(original_data, embed1, k = k)
  trust2 <- trustworthiness(original_data, embed2, k = k)
  dist_cor <- distance_correlation(embed1, embed2)

  list(
    trust_embedmit = trust1,
    trust_embed = trust2,
    trust_diff = abs(trust1 - trust2),
    distance_correlation = dist_cor
  )
}

# Create test data for recipe steps
create_recipe_test_data <- function(n = 500, seed = 42) {
  set.seed(seed)

  # Create data with various column types suitable for recipe steps
  data.frame(
    # Numeric predictors
    x1 = rnorm(n),
    x2 = rnorm(n, mean = 5),
    x3 = runif(n, 0, 10),
    x4 = rexp(n, rate = 0.5),

    # Categorical predictors with different cardinalities
    cat_low = factor(sample(LETTERS[1:3], n, replace = TRUE)),
    cat_med = factor(sample(letters[1:10], n, replace = TRUE)),
    cat_high = factor(sample(paste0("level_", 1:15), n, replace = TRUE)),

    # Numeric outcome (for regression)
    outcome_num = rnorm(n, mean = 10),

    # Categorical outcome (for classification)
    outcome_cat = factor(sample(c("class_A", "class_B", "class_C"), n, replace = TRUE)),

    stringsAsFactors = FALSE
  )
}

# Create binned test data for WOE and similar steps
create_binned_test_data <- function(n = 500, seed = 42) {
  set.seed(seed)

  # Create data with relationships suitable for WOE
  prob <- plogis(rnorm(n))
  outcome <- rbinom(n, 1, prob)

  data.frame(
    x1 = rnorm(n) + outcome,
    x2 = rnorm(n) + 0.5 * outcome,
    x3 = rnorm(n),
    cat1 = factor(sample(LETTERS[1:5], n, replace = TRUE, prob = c(0.1, 0.2, 0.3, 0.25, 0.15))),
    cat2 = factor(sample(letters[1:8], n, replace = TRUE)),
    outcome = factor(outcome, levels = c(0, 1), labels = c("no", "yes")),
    stringsAsFactors = FALSE
  )
}

# Create UMAP test data
create_umap_test_data <- function(n = 150, p = 4, seed = 42) {
  set.seed(seed)

  # Use iris-like structure with clear clusters
  n_per_cluster <- n %/% 3
  remainder <- n %% 3

  cluster1 <- matrix(rnorm(n_per_cluster * p, mean = 0), ncol = p)
  cluster2 <- matrix(rnorm(n_per_cluster * p, mean = 3), ncol = p)
  cluster3 <- matrix(rnorm((n_per_cluster + remainder) * p, mean = -3), ncol = p)

  X <- rbind(cluster1, cluster2, cluster3)
  colnames(X) <- paste0("x", seq_len(p))

  outcome <- factor(c(
    rep("A", n_per_cluster),
    rep("B", n_per_cluster),
    rep("C", n_per_cluster + remainder)
  ))

  data.frame(X, outcome = outcome)
}

# Format comparison results for test output
format_exact_comparison <- function(comparison_result, labels = c("embedmit", "embed")) {
  if (comparison_result$equivalent) {
    sprintf("Results are equivalent (max diff: %.2e)", comparison_result$max_diff)
  } else {
    sprintf(
      "Results differ:\n  Max diff: %.2e\n  Differing columns: %s",
      comparison_result$max_diff,
      paste(comparison_result$diff_cols, collapse = ", ")
    )
  }
}

format_umap_comparison <- function(comparison_result, labels = c("embedmit", "embed")) {
  sprintf(
    "UMAP Comparison:\n  Trustworthiness: %s=%.4f, %s=%.4f (diff=%.4f)\n  Distance Correlation: %.4f",
    labels[1], comparison_result$trust_embedmit,
    labels[2], comparison_result$trust_embed,
    comparison_result$trust_diff,
    comparison_result$distance_correlation
  )
}
