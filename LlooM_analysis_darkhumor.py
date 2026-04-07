import os

import numpy as np
import pandas as pd
import text_lloom.workbench as wb
from text_lloom.models import OpenAIEmbedModel, OpenAIModel

pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_rows", None)

openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise EnvironmentError("OPENAI_API_KEY is not set.")

DATA_PATH = "unified_dataset.csv"
df = pd.read_csv(DATA_PATH, dtype=str)

l = wb.lloom(
    df=df,
    text_col="description",
    id_col="video_id",
    distill_model=OpenAIModel(
        name="gpt-4o-mini",
        api_key=openai_key,
        context_window=128_000,
        cost=(0.15 / 1_000_000, 0.6 / 1_000_000),
        rate_limit=(20, 10),
    ),
    cluster_model=OpenAIEmbedModel(
        name="text-embedding-3-large",
        api_key=openai_key,
    ),
    synth_model=OpenAIModel(
        name="gpt-4o",
        api_key=openai_key,
        context_window=128_000,
        cost=(2.5 / 1_000_000, 10 / 1_000_000),
        rate_limit=(20, 10),
    ),
    score_model=OpenAIModel(
        name="gpt-4o-mini",
        api_key=openai_key,
        context_window=128_000,
        cost=(0.15 / 1_000_000, 0.6 / 1_000_000),
        rate_limit=(20, 10),
    ),
)
