# Run Comparison Tests

Run the comparison test suite that compares embedmit against the original embed package.

```bash
cd /Users/rmsharp/Documents/R_packages/embedmit && Rscript -e "testthat::test_file('tests/testthat/test_comparison_embed.R')"
```

After running, report whether all comparison tests passed and highlight any differences found between embedmit and embed.
