# embedmit 1.0.0

## Initial Release

This is the first release of embedmit, an MIT-compatible fork of the embed
package (version 1.2.1) by Emil Hvitfeldt and Max Kuhn (Posit Software, PBC).

### Key Differences from embed

* **MIT-Compatible**: embedmit depends on uwotmit instead of uwot for UMAP
  functionality. This avoids the AGPL-licensed dqrng dependency that uwot uses,
  making embedmit suitable for inclusion in projects requiring permissive
  licensing.

* **Default RNG**: The `step_umap()` function defaults to `rng_type = "tausworthe"`
  instead of `"pcg"` to avoid the AGPL-licensed PCG implementation. Users can
  still explicitly request `rng_type = "pcg"` if desired, which will use the
  MIT-licensed sitmo implementation from uwotmit.

* **Package renamed**: The package is named `embedmit` to distinguish it from
  the original `embed` package.

### Inherited Features from embed 1.2.1

embedmit inherits all features from embed 1.2.1, including:

* `step_umap()` for UMAP dimensionality reduction
* `step_lencode_glm()`, `step_lencode_bayes()`, `step_lencode_mixed()` for
  likelihood encodings
* `step_lencode()` for analytical likelihood encoding with optional smoothing
* `step_woe()` for weight of evidence encodings
* `step_embed()` for entity embeddings using neural networks
* `step_discretize_cart()` and `step_discretize_xgb()` for supervised binning
* `step_collapse_cart()` and `step_collapse_stringdist()` for factor level pooling
* `step_pca_sparse()`, `step_pca_sparse_bayes()`, and `step_pca_truncated()` for
  sparse PCA methods
* Full integration with the tidymodels ecosystem

### Acknowledgments

This package is based on embed by Emil Hvitfeldt, Max Kuhn, and Posit Software,
PBC. All credit for the recipe step implementations goes to the original
authors. This fork exists solely to provide an MIT-compatible alternative for
users who require permissive licensing.

For the original embed package, see: https://github.com/tidymodels/embed
