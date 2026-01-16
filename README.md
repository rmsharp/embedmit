# embedmit

<!-- badges: start -->
[![R-CMD-check](https://github.com/rmsharp/embedmit/workflows/R-CMD-check/badge.svg)](https://github.com/rmsharp/embedmit/actions)
<!-- badges: end -->

## MIT-Compatible Fork of embed

`embedmit` is an MIT-licensed fork of the [embed](https://github.com/tidymodels/embed) package from tidymodels. It provides the same recipe steps for encoding predictors but avoids AGPL-licensed dependencies at runtime.

### Why embedmit?

The original `embed` package depends on `uwot` for UMAP functionality, which in turn depends on `dqrng` (AGPL-3 licensed). This creates licensing complications for projects that need to maintain MIT or other permissive licensing throughout their dependency tree.

`embedmit` modifies the default options in `step_umap()` to use `rng_type = "tausworthe"` instead of the default PCG random number generator. The tausworthe RNG is built into uwot and doesn't require the AGPL-licensed dqrng package.

### Key Differences from embed

| Feature | embed | embedmit |
|---------|-------|----------|
| License | MIT | MIT |
| Default RNG for UMAP | PCG (via dqrng, AGPL-3) | Tausworthe (built-in, no AGPL) |
| Functionality | Full | Full |

### Technical Change

The only code change is in `R/umap.R`:

```r
# embed (original)
options = list(verbose = FALSE, n_threads = 1)

# embedmit
options = list(verbose = FALSE, n_threads = 1, rng_type = "tausworthe")
```

This ensures that when `step_umap()` calls uwot, it uses the tausworthe random number generator instead of PCG, avoiding the dqrng dependency at runtime.

## Installing

### From GitHub

```R
# install.packages("devtools")
devtools::install_github("rmsharp/embedmit")
```

## Introduction

`embedmit` has extra steps for the [`recipes`](https://recipes.tidymodels.org/) package for embedding predictors into one or more numeric columns. Almost all of the preprocessing methods are *supervised*.

These steps are available here in a separate package because the step dependencies, [`rstanarm`](https://CRAN.r-project.org/package=rstanarm), [`lme4`](https://CRAN.r-project.org/package=lme4), and [`keras3`](https://CRAN.r-project.org/package=keras3), are fairly heavy.

### Available Steps

**For categorical predictors:**

- `step_lencode_glm()`, `step_lencode_bayes()`, and `step_lencode_mixed()` - effect encoding using generalized linear models
- `step_embed()` - neural network embeddings using keras3
- `step_woe()` - weight of evidence encodings

**For numeric predictors:**

- `step_umap()` - UMAP dimensionality reduction (using tausworthe RNG by default)
- `step_discretize_xgb()` and `step_discretize_cart()` - supervised binning
- `step_pca_sparse()` and `step_pca_sparse_bayes()` - sparse PCA

## Example

```R
library(embedmit)
library(recipes)

# Create a recipe with UMAP embedding
rec <- recipe(Species ~ ., data = iris) |>
  step_normalize(all_numeric_predictors()) |>
  step_umap(all_numeric_predictors(), num_comp = 2)

# Prepare and bake
prepped <- prep(rec)
baked <- bake(prepped, new_data = NULL)
```

## Documentation

For detailed documentation on the embedding steps, please refer to the original embed documentation at <https://embed.tidymodels.org/>.

## License

[MIT](LICENSE.md)

## Acknowledgments

This package is a fork of [embed](https://github.com/tidymodels/embed) by Emil Hvitfeldt and Max Kuhn (Posit). All credit for the core functionality goes to the original authors and contributors.

## See Also

* The original [embed package](https://github.com/tidymodels/embed)
* [uwotlite](https://github.com/rmsharp/uwotlite) - MIT-compatible fork of uwot with sitmo instead of dqrng
* The [tidymodels](https://www.tidymodels.org/) framework
