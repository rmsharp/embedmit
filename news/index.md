# Changelog

## embedmit 1.0.0

### Initial Release

This is the first release of embedmit, an MIT-compatible fork of the
embed package (version 1.2.1) by Emil Hvitfeldt and Max Kuhn (Posit
Software, PBC).

#### Key Differences from embed

- **MIT-Compatible**: embedmit depends on uwotmit instead of uwot for
  UMAP functionality. This avoids the AGPL-licensed dqrng dependency
  that uwot uses, making embedmit suitable for inclusion in projects
  requiring permissive licensing.

- **Default RNG**: The
  [`step_umap()`](https://rmsharp.github.io/embedmit/reference/step_umap.md)
  function defaults to `rng_type = "tausworthe"` instead of `"pcg"` to
  avoid the AGPL-licensed PCG implementation. Users can still explicitly
  request `rng_type = "pcg"` if desired, which will use the MIT-licensed
  sitmo implementation from uwotmit.

- **Package renamed**: The package is named `embedmit` to distinguish it
  from the original `embed` package.

#### Inherited Features from embed 1.2.1

embedmit inherits all features from embed 1.2.1, including:

- [`step_umap()`](https://rmsharp.github.io/embedmit/reference/step_umap.md)
  for UMAP dimensionality reduction
- [`step_lencode_glm()`](https://rmsharp.github.io/embedmit/reference/step_lencode_glm.md),
  [`step_lencode_bayes()`](https://rmsharp.github.io/embedmit/reference/step_lencode_bayes.md),
  [`step_lencode_mixed()`](https://rmsharp.github.io/embedmit/reference/step_lencode_mixed.md)
  for likelihood encodings
- [`step_lencode()`](https://rmsharp.github.io/embedmit/reference/step_lencode.md)
  for analytical likelihood encoding with optional smoothing
- [`step_woe()`](https://rmsharp.github.io/embedmit/reference/step_woe.md)
  for weight of evidence encodings
- [`step_embed()`](https://rmsharp.github.io/embedmit/reference/step_embed.md)
  for entity embeddings using neural networks
- [`step_discretize_cart()`](https://rmsharp.github.io/embedmit/reference/step_discretize_cart.md)
  and
  [`step_discretize_xgb()`](https://rmsharp.github.io/embedmit/reference/step_discretize_xgb.md)
  for supervised binning
- [`step_collapse_cart()`](https://rmsharp.github.io/embedmit/reference/step_collapse_cart.md)
  and
  [`step_collapse_stringdist()`](https://rmsharp.github.io/embedmit/reference/step_collapse_stringdist.md)
  for factor level pooling
- [`step_pca_sparse()`](https://rmsharp.github.io/embedmit/reference/step_pca_sparse.md),
  [`step_pca_sparse_bayes()`](https://rmsharp.github.io/embedmit/reference/step_pca_sparse_bayes.md),
  and
  [`step_pca_truncated()`](https://rmsharp.github.io/embedmit/reference/step_pca_truncated.md)
  for sparse PCA methods
- Full integration with the tidymodels ecosystem

#### Acknowledgments

This package is based on embed by Emil Hvitfeldt, Max Kuhn, and Posit
Software, PBC. All credit for the recipe step implementations goes to
the original authors. This fork exists solely to provide an
MIT-compatible alternative for users who require permissive licensing.

For the original embed package, see:
<https://github.com/tidymodels/embed>
