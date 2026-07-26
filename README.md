# Tracking Political Change with Embeddings of Parliamentary Speech

Embedding parliamentary speeches to trace MP/party trajectories over time, validated against political scores.

## Project Overview

This project uses encoder-only language models to build vector representations of parliamentary speeches and the MPs who give them, then studies how those representations evolve over time to trace ideological trajectories. Because an embedding space is only useful if its geometry is politically meaningful, the focus is validation: testing whether the representations recover known structure such as party membership, survey ideology scores, and real political events.

## Data

The raw datasets are **not included in this repository** (they are large and, in the case of CHES and the Manifesto Project, subject to the providers' terms of use). Download them as described below and place them in `data/raw/`.

- **Open Discourse (Bundestag speeches)** — download from the Open Discourse Harvard Dataverse (linked from the Open Discourse website). Use the speeches file (`speeches.csv`); the accompanying `factions.csv` is also needed to map faction IDs to party names.
- **Chapel Hill Expert Survey (CHES)** — download the latest version from the CHES website (the most recent European edition covers data up to 2024).
- **Manifesto Project (CMP/MARPOR)** — download the Main Dataset (edition 2025a) from the Manifesto Project website. Registration/login is required. Make sure to download the *Main Dataset* (CSV, 100+ columns including `rile`), not one of the auxiliary party-list files.

After downloading, your `data/raw/` folder should contain the speech corpus, the factions table, the CHES file, and the Manifesto Main Dataset.

## Repo Structure

- `/data` — `data/raw/` holds the downloaded datasets (git-ignored); `data/processed/` holds cleaned outputs
- `/notebooks` — analysis notebooks (data cleaning, preprocessing, embeddings, validation)
