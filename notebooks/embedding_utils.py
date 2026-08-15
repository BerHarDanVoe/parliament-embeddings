"""
Shared embedding helpers for the parliamentary-speech-embeddings pipeline.
Used by 02-01_embedding_baseline.ipynb and 02-02_finetuning_embedding.ipynb
so both notebooks embed speeches the same way and can't silently drift apart
(e.g. different max_length or pooling logic between the two).
 
Place this file in the same directory as the notebooks and import with:
    from embedding_utils import mean_pooling, embed_batch, embed_speeches
"""
import numpy as np
import torch
import torch.nn.functional as F


def mean_pooling(last_hidden_state, attention_mask):
    """Average token embeddings into one vector per speech, ignoring padding."""
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)  # avoid divide-by-zero
    return summed / counts

def embed_speeches(
    texts,
    model,
    tokenizer,
    device,
    batch_size=32,
    max_length=8192,
    checkpoint_every=200,
    checkpoint_path=None,
):
    """Embeds a list of raw speech strings in batches (full corpus too large
    for one pass). Used for full-corpus embedding runs (baseline + fine-tuned).
 
    max_length default (8192) is Jina v3's max context, which comfortably
    clears the 100-8000 token cleaning filter used upstream on this corpus.
    checkpoint_path: if given, periodically saves progress to this .npy file
    as a crash safety net (not currently used to resume a run).
    """
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            emb = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
            emb = F.normalize(emb, p=2, dim=1)
        all_embeddings.append(emb.cpu().numpy())
        print(f"Embedded {i + len(batch)}/{len(texts)}")
 
        batch_num = i // batch_size
        if checkpoint_path and batch_num % checkpoint_every == 0 and batch_num > 0:
            np.save(checkpoint_path, np.vstack(all_embeddings))
            print(f"  checkpoint saved at {i + len(batch)} speeches")
 
    return np.vstack(all_embeddings)