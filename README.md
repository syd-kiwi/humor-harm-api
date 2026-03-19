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

## Agreement score analyzer
Use `scripts/annotations/agreement_scores.py` to export disagreement, Cohen's kappa, and Krippendorff's alpha outputs for annotation datasets:

```bash
python scripts/annotations/agreement_scores.py <csv_path> --item_col id --annotator_col annotator
```

The script now writes:
- `<stem>_worst_ids_overall_top<k>.csv` for the highest-disagreement items
- `<stem>_cohen_kappa_scores.csv` with pairwise Cohen's kappa by label and annotator pair
- `<stem>_cohen_kappa_summary.csv` with per-label aggregate kappa stats plus Krippendorff's alpha
- `<stem>_videos_used_for_kappa.csv` listing the item/video rows used in each kappa calculation

If your input has metadata columns like `video_id`, `url`, and `title`, those are included automatically in the saved `videos_used` file and in the console preview.

## Hashtag popularity analyzer
Use `scripts/analyze_hashtags.py` to find the most popular hashtags in your dataset:

```bash
python scripts/analyze_hashtags.py unified_dataset.csv --top 30 --output hashtag_counts.csv
```

By default it scans `description`, `tags`, and `title` columns for hashtag tokens (like `#funny`).
