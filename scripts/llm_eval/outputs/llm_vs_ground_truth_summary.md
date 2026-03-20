# LLM vs Ground Truth Summary

- Ground truth file: `annotation_dashboard/03-18.json`
- LLM output directory: `scripts/llm_eval/outputs`
- Ground truth items loaded: **1211**
- LLM output files compared: **5**

## Overall ranking

| Rank | Model | Field-level accuracy | Matches | Evaluated |
|---|---|---:|---:|---:|
| 1 | gemini-3-flash-preview | 60.80% | 5154 | 8477 |
| 2 | gpt-5-mini-2025-08-07 | 59.07% | 5007 | 8477 |
| 3 | claude-sonnet-4-6 | 58.44% | 4954 | 8477 |
| 4 | gemini-2.5-flash | 57.25% | 4853 | 8477 |
| 5 | deepseek-v3.1 | 56.94% | 4827 | 8477 |

## claude-sonnet-4-6

| Field | Accuracy | Matches | Evaluated | Top mismatches |
|---|---:|---:|---:|---|
| __overall__ | 58.44% | 4954 | 8477 |  |
| humor_presence | 50.12% | 607 | 1211 | 245x GT=Not Humor || PRED=Humor ; 174x GT=Not Humor || PRED=Almost or not sure ; 69x GT=Almost or not sure || PRED=Humor ; 60x GT=Humor || PRED=Almost or not sure ; 33x GT=Humor || PRED=Not Humor |
| joke_topic | 20.56% | 249 | 1211 | 104x GT=Other || PRED=[] ; 68x GT=Celebrity and pop culture || PRED=Other ; 41x GT=Celebrity and pop culture || PRED=Celebrity and pop culture | Other ; 36x GT=Other || PRED=Celebrity and pop culture | Other ; 29x GT=Other || PRED=Other | Relationships |
| rhetorical_device | 57.14% | 692 | 1211 | 114x GT=Satire || PRED=None of these ; 112x GT=None of these || PRED=Irony ; 66x GT=Irony || PRED=None of these ; 61x GT=None of these || PRED=Irony | Satire ; 45x GT=Satire || PRED=Irony | Satire |
| stand_up | 82.00% | 993 | 1211 | 205x GT=No || PRED=Yes ; 11x GT=Yes || PRED=No ; 2x GT= || PRED=No |
| humor_type | 45.83% | 555 | 1211 | 380x GT=None of these || PRED=Regular Humor ; 109x GT=Dark Humor || PRED=Regular Humor ; 73x GT=None of these || PRED=Dark Humor ; 46x GT=Regular Humor || PRED=Dark Humor ; 45x GT=Regular Humor || PRED=None of these |
| target_category | 76.63% | 928 | 1211 | 30x GT=Race / Ethnicity || PRED=[] ; 23x GT=Gender / Sex related || PRED=[] ; 20x GT=Other Sensitive Target || PRED=[] ; 19x GT=[] || PRED=Violence / Death ; 18x GT=[] || PRED=Other Sensitive Target | Violence / Death |
| dark_intensity | 76.80% | 930 | 1211 | 97x GT= || PRED=2 - Moderate ; 80x GT=1 - Mild || PRED= ; 42x GT=1 - Mild || PRED=2 - Moderate ; 24x GT=2 - Moderate || PRED= ; 13x GT= || PRED=3 - Severe |

## deepseek-v3.1

| Field | Accuracy | Matches | Evaluated | Top mismatches |
|---|---:|---:|---:|---|
| __overall__ | 56.94% | 4827 | 8477 |  |
| humor_presence | 56.40% | 683 | 1211 | 321x GT=Not Humor || PRED=Humor ; 81x GT=Almost or not sure || PRED=Humor ; 66x GT=Humor || PRED=Not Humor ; 46x GT=Almost or not sure || PRED=Not Humor ; 9x GT=Not Humor || PRED=Almost or not sure |
| joke_topic | 20.15% | 244 | 1211 | 171x GT=Other || PRED=[] ; 68x GT=Other || PRED=Celebrity and pop culture ; 63x GT=Celebrity and pop culture || PRED=[] ; 42x GT=Politics and society || PRED=[] ; 38x GT=Other || PRED=Celebrity and pop culture | Relationships |
| rhetorical_device | 48.89% | 592 | 1211 | 240x GT=None of these || PRED=Irony ; 101x GT=Satire || PRED=Irony ; 66x GT=Satire || PRED=None of these ; 53x GT=Irony | Satire || PRED=Irony ; 44x GT=Irony || PRED=None of these |
| stand_up | 68.13% | 825 | 1211 | 364x GT=No || PRED=Yes ; 20x GT=Yes || PRED=No ; 1x GT= || PRED=Yes ; 1x GT= || PRED=No |
| humor_type | 50.21% | 608 | 1211 | 280x GT=None of these || PRED=Regular Humor ; 121x GT=Dark Humor || PRED=Regular Humor ; 86x GT=Regular Humor || PRED=None of these ; 74x GT=None of these || PRED=Dark Humor ; 32x GT=Regular Humor || PRED=Dark Humor |
| target_category | 77.79% | 942 | 1211 | 39x GT=Race / Ethnicity || PRED=[] ; 33x GT=[] || PRED=Violence / Death ; 26x GT=Gender / Sex related || PRED=[] ; 20x GT=Other Sensitive Target || PRED=[] ; 16x GT=[] || PRED=Crime | Violence / Death |
| dark_intensity | 77.04% | 933 | 1211 | 98x GT=1 - Mild || PRED= ; 68x GT= || PRED=2 - Moderate ; 27x GT=1 - Mild || PRED=2 - Moderate ; 26x GT=2 - Moderate || PRED= ; 20x GT= || PRED=3 - Severe |

## gemini-2.5-flash

| Field | Accuracy | Matches | Evaluated | Top mismatches |
|---|---:|---:|---:|---|
| __overall__ | 57.25% | 4853 | 8477 |  |
| humor_presence | 53.43% | 647 | 1211 | 391x GT=Not Humor || PRED=Humor ; 97x GT=Almost or not sure || PRED=Humor ; 39x GT=Humor || PRED=Not Humor ; 29x GT=Almost or not sure || PRED=Not Humor ; 4x GT=Not Humor || PRED=Almost or not sure |
| joke_topic | 27.66% | 335 | 1211 | 111x GT=Other || PRED=[] ; 84x GT=Celebrity and pop culture || PRED=Other ; 41x GT=Celebrity and pop culture || PRED=[] ; 36x GT=Other || PRED=Politics and society ; 28x GT=Politics and society || PRED=[] |
| rhetorical_device | 52.77% | 639 | 1211 | 159x GT=None of these || PRED=Irony ; 88x GT=Satire || PRED=None of these ; 68x GT=None of these || PRED=Irony | Satire ; 55x GT=Satire || PRED=Irony ; 50x GT=Irony || PRED=None of these |
| stand_up | 73.16% | 886 | 1211 | 313x GT=No || PRED=Yes ; 10x GT=Yes || PRED=No ; 1x GT= || PRED=No ; 1x GT= || PRED=Yes |
| humor_type | 45.75% | 554 | 1211 | 341x GT=None of these || PRED=Regular Humor ; 108x GT=Dark Humor || PRED=Regular Humor ; 95x GT=None of these || PRED=Dark Humor ; 56x GT=Regular Humor || PRED=None of these ; 54x GT=Regular Humor || PRED=Dark Humor |
| target_category | 74.57% | 903 | 1211 | 44x GT=[] || PRED=Violence / Death ; 32x GT=Race / Ethnicity || PRED=[] ; 19x GT=Gender / Sex related || PRED=[] ; 18x GT=[] || PRED=Crime | Violence / Death ; 17x GT=Other Sensitive Target || PRED=[] |
| dark_intensity | 73.41% | 889 | 1211 | 83x GT= || PRED=2 - Moderate ; 80x GT=1 - Mild || PRED= ; 50x GT= || PRED=3 - Severe ; 28x GT=1 - Mild || PRED=2 - Moderate ; 24x GT=2 - Moderate || PRED= |

## gemini-3-flash-preview

| Field | Accuracy | Matches | Evaluated | Top mismatches |
|---|---:|---:|---:|---|
| __overall__ | 60.80% | 5154 | 8477 |  |
| humor_presence | 57.56% | 697 | 1211 | 323x GT=Not Humor || PRED=Humor ; 87x GT=Almost or not sure || PRED=Humor ; 45x GT=Humor || PRED=Not Humor ; 32x GT=Almost or not sure || PRED=Not Humor ; 18x GT=Not Humor || PRED=Almost or not sure |
| joke_topic | 22.79% | 276 | 1211 | 154x GT=Other || PRED=[] ; 51x GT=Celebrity and pop culture || PRED=Other ; 49x GT=Celebrity and pop culture || PRED=[] ; 35x GT=Politics and society || PRED=[] ; 28x GT=Other || PRED=Celebrity and pop culture |
| rhetorical_device | 52.68% | 638 | 1211 | 149x GT=None of these || PRED=Irony ; 71x GT=None of these || PRED=Irony | Satire ; 64x GT=Satire || PRED=None of these ; 58x GT=Satire || PRED=Irony ; 51x GT=None of these || PRED=Satire |
| stand_up | 84.97% | 1029 | 1211 | 168x GT=No || PRED=Yes ; 12x GT=Yes || PRED=No ; 2x GT= || PRED=No |
| humor_type | 49.96% | 605 | 1211 | 317x GT=None of these || PRED=Regular Humor ; 132x GT=Dark Humor || PRED=Regular Humor ; 61x GT=None of these || PRED=Dark Humor ; 60x GT=Regular Humor || PRED=None of these ; 29x GT=Regular Humor || PRED=Dark Humor |
| target_category | 79.27% | 960 | 1211 | 38x GT=Race / Ethnicity || PRED=[] ; 28x GT=[] || PRED=Violence / Death ; 25x GT=Gender / Sex related || PRED=[] ; 21x GT=Other Sensitive Target || PRED=[] ; 16x GT=Violence / Death || PRED=[] |
| dark_intensity | 78.36% | 949 | 1211 | 100x GT=1 - Mild || PRED= ; 41x GT= || PRED=2 - Moderate ; 39x GT= || PRED=1 - Mild ; 32x GT=2 - Moderate || PRED= ; 21x GT=1 - Mild || PRED=2 - Moderate |

## gpt-5-mini-2025-08-07

| Field | Accuracy | Matches | Evaluated | Top mismatches |
|---|---:|---:|---:|---|
| __overall__ | 59.07% | 5007 | 8477 |  |
| humor_presence | 54.17% | 656 | 1211 | 329x GT=Not Humor || PRED=Humor ; 84x GT=Almost or not sure || PRED=Humor ; 56x GT=Humor || PRED=Not Humor ; 34x GT=Almost or not sure || PRED=Not Humor ; 29x GT=Not Humor || PRED=Almost or not sure |
| joke_topic | 20.97% | 254 | 1211 | 144x GT=Other || PRED=[] ; 45x GT=Celebrity and pop culture || PRED=[] ; 42x GT=Other || PRED=Celebrity and pop culture ; 42x GT=Politics and society || PRED=[] ; 40x GT=Celebrity and pop culture || PRED=Other |
| rhetorical_device | 57.14% | 692 | 1211 | 139x GT=None of these || PRED=Irony ; 94x GT=Satire || PRED=None of these ; 63x GT=Satire || PRED=Irony ; 60x GT=Irony || PRED=None of these ; 37x GT=Irony | Satire || PRED=Irony |
| stand_up | 74.40% | 901 | 1211 | 298x GT=No || PRED=Yes ; 10x GT=Yes || PRED=No ; 2x GT= || PRED=No |
| humor_type | 46.90% | 568 | 1211 | 342x GT=None of these || PRED=Regular Humor ; 152x GT=Dark Humor || PRED=Regular Humor ; 71x GT=Regular Humor || PRED=None of these ; 46x GT=None of these || PRED=Dark Humor ; 26x GT=Regular Humor || PRED=Dark Humor |
| target_category | 80.68% | 977 | 1211 | 51x GT=Race / Ethnicity || PRED=[] ; 41x GT=[] || PRED=Violence / Death ; 28x GT=Gender / Sex related || PRED=[] ; 21x GT=Other Sensitive Target || PRED=[] ; 13x GT=Violence / Death || PRED=[] |
| dark_intensity | 79.19% | 959 | 1211 | 113x GT=1 - Mild || PRED= ; 36x GT=2 - Moderate || PRED= ; 34x GT= || PRED=2 - Moderate ; 21x GT= || PRED=3 - Severe ; 18x GT= || PRED=1 - Mild |
