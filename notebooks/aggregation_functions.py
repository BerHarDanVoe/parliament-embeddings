import numpy as np
import pandas as pd

def aggregate_by_mp_year(data):
    """Aggregates speech-level embeddings to one embedding per (politicianId, year), by averaging."""
    agg_records = []
    for (politicianId, year), idx in data.groupby(["politicianId", "year"]).groups.items():
        mean_embedding = np.mean(np.stack(data.loc[idx, "embedding"]), axis=0)
        agg_records.append({
            "politicianId": politicianId,
            "year": year,
            "party": data.loc[idx, "party"].iloc[0],
            "n_speeches": len(idx),
            "embedding": mean_embedding
        })
    return pd.DataFrame(agg_records)


def aggregate_by_party_year(data):
    """Aggregates speech-level embeddings to one embedding per (party, year), by averaging."""
    agg_records = []
    for (party, year), idx in data.groupby(["party", "year"]).groups.items():
        mean_embedding = np.mean(np.stack(data.loc[idx, "embedding"]), axis=0)
        agg_records.append({
            "year": year,
            "party": data.loc[idx, "party"].iloc[0],
            "n_speeches": len(idx),
            "embedding": mean_embedding
        })
    return pd.DataFrame(agg_records)