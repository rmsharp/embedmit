# embedmit

## MIT-Compatible Fork of embed

`embedmit` is an MIT-licensed fork of the
[embed](https://github.com/tidymodels/embed) package from tidymodels. It
provides the same recipe steps for encoding predictors but avoids
AGPL-licensed dependencies at runtime.

### Why embedmit?

The original `embed` package depends on `uwot` for UMAP functionality,
which in turn depends on `dqrng` (AGPL-3 licensed). This creates
licensing complications for projects that need to maintain MIT or other
permissive licensing throughout their dependency tree.

`embedmit` solves this by: 1. Depending on
[uwotmit](https://github.com/rmsharp/uwotmit) instead of `uwot` - an
MIT-licensed fork that replaces `dqrng` with `sitmo` 2. Using
`rng_type = "tausworthe"` as the default for additional safety

### Key Differences from embed

| Feature              | embed                         | embedmit                |
|----------------------|-------------------------------|-------------------------|
| License              | MIT                           | MIT                     |
| UMAP dependency      | uwot (requires dqrng, AGPL-3) | uwotmit (MIT-only deps) |
| Default RNG for UMAP | PCG (via dqrng)               | Tausworthe (built-in)   |
| Functionality        | Full                          | Full                    |

### Technical Changes

1.  **Dependency change**: `uwot` → `uwotmit` in DESCRIPTION
2.  **Default RNG** in `R/umap.R`:

``` r
# embed (original)
options = list(verbose = FALSE, n_threads = 1)

# embedmit
options = list(verbose = FALSE, n_threads = 1, rng_type = "tausworthe")
```

This ensures a fully MIT-compatible dependency chain with no
AGPL-licensed packages.

## Installation

### From GitHub

``` r
# install.packages("devtools")
devtools::install_github("rmsharp/embedmit")
```

Note that to use some steps, you will also have to install other
packages such as `rstanarm` and `lme4`. For all of the steps to work,
you may want to use:

``` r
install.packages(c("rpart", "xgboost", "rstanarm", "lme4"))
```

## Introduction

`embedmit` has extra steps for the
[`recipes`](https://recipes.tidymodels.org/) package for embedding
predictors into one or more numeric columns. Almost all of the
preprocessing methods are *supervised*.

These steps are available here in a separate package because the step
dependencies, [`rstanarm`](https://CRAN.r-project.org/package=rstanarm),
[`lme4`](https://CRAN.r-project.org/package=lme4), and
[`keras3`](https://CRAN.r-project.org/package=keras3), are fairly heavy.

### Available Steps

**For categorical predictors:**

- [`step_lencode_glm()`](https://rmsharp.github.io/embedmit/dev/reference/step_lencode_glm.md),
  [`step_lencode_bayes()`](https://rmsharp.github.io/embedmit/dev/reference/step_lencode_bayes.md),
  and
  [`step_lencode_mixed()`](https://rmsharp.github.io/embedmit/dev/reference/step_lencode_mixed.md)
  estimate the effect of each of the factor levels on the outcome and
  these estimates are used as the new encoding. The estimates are
  estimated by a generalized linear model. This step can be executed
  without pooling (via `glm`) or with partial pooling (`stan_glm` or
  `lmer`). Currently implemented for numeric and two-class outcomes.

- [`step_embed()`](https://rmsharp.github.io/embedmit/dev/reference/step_embed.md)
  uses
  [`keras3::layer_embedding`](https://keras3.posit.co/reference/layer_embedding.html)
  to translate the original *C* factor levels into a set of *D* new
  variables (\< *C*). The model fitting routine optimizes which factor
  levels are mapped to each of the new variables as well as the
  corresponding regression coefficients (i.e., neural network weights)
  that will be used as the new encodings.

- [`step_woe()`](https://rmsharp.github.io/embedmit/dev/reference/step_woe.md)
  creates new variables based on weight of evidence encodings.

**For numeric predictors:**

- [`step_umap()`](https://rmsharp.github.io/embedmit/dev/reference/step_umap.md)
  uses a nonlinear transformation similar to t-SNE but can be used to
  project the transformation on new data. Both supervised and
  unsupervised methods can be used. **Note:** In embedmit, this uses
  [uwotmit](https://github.com/rmsharp/uwotmit) and defaults to the
  tausworthe RNG for a fully MIT-compatible dependency chain.

- [`step_discretize_xgb()`](https://rmsharp.github.io/embedmit/dev/reference/step_discretize_xgb.md)
  and
  [`step_discretize_cart()`](https://rmsharp.github.io/embedmit/dev/reference/step_discretize_cart.md)
  can make binned versions of numeric predictors using supervised
  tree-based models.

- [`step_pca_sparse()`](https://rmsharp.github.io/embedmit/dev/reference/step_pca_sparse.md)
  and
  [`step_pca_sparse_bayes()`](https://rmsharp.github.io/embedmit/dev/reference/step_pca_sparse_bayes.md)
  conduct feature extraction with sparsity of the component loadings.

## Example

``` r
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

For detailed documentation on the embedding steps, please refer to the
original embed documentation at <https://embed.tidymodels.org/>.

Some references for these methods are:

- Francois C and Allaire JJ (2018) [*Deep Learning with
  R*](https://www.manning.com/books/deep-learning-with-r), Manning
- Guo, C and Berkhahn F (2016) “[Entity Embeddings of Categorical
  Variables](https://arxiv.org/abs/1604.06737)”
- Micci-Barreca D (2001) “[A preprocessing scheme for high-cardinality
  categorical attributes in classification and prediction
  problems](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=A+preprocessing+scheme+for+high-cardinality+categorical+attributes+in+classification+and+prediction+problems&btnG=),”
  ACM SIGKDD Explorations Newsletter, 3(1), 27-32.
- Zumel N and Mount J (2017) “[`vtreat`: a `data.frame` Processor for
  Predictive Modeling](https://arxiv.org/abs/1611.09477)”
- McInnes L and Healy J (2018) [UMAP: Uniform Manifold Approximation and
  Projection for Dimension Reduction](https://arxiv.org/abs/1802.03426)
- Good, I. J. (1985), “[Weight of evidence: A brief
  survey](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Weight+of+evidence%3A+A+brief+survey&btnG=)”,
  Bayesian Statistics, 2, pp.249-270.

## License

[MIT](https://rmsharp.github.io/embedmit/dev/LICENSE.md)

## Acknowledgments

This package is a fork of [embed](https://github.com/tidymodels/embed)
by Emil Hvitfeldt and Max Kuhn (Posit). All credit for the core
functionality goes to the original authors and contributors.

## See Also

- The original [embed package](https://github.com/tidymodels/embed)
- [uwotmit](https://github.com/rmsharp/uwotmit) - MIT-compatible fork of
  uwot with sitmo instead of dqrng
- The [tidymodels](https://www.tidymodels.org/) framework
