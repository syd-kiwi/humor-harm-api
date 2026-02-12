# humor-harm-api
Repository for API scripts, Label Studio configurations, and annotation pipeline setup for detecting humor, dark humor, and harmful content in short-form videos.

## Cohen's Kappa output helper
Use `cohen_kappa_scores.py` when you only want Cohen's Kappa outputs:

```bash
python cohen_kappa_scores.py <csv_path> <rater1_column> <rater2_column>
```

The script prints exactly two results:
- Cohen's Kappa for the first 50 paired scores (or fewer if the file has < 50 rows)
- Cohen's Kappa for all paired scores
