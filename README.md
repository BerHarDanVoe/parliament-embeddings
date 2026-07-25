# Tracking Political Change with Embeddings of Parliamentary Speech

Embedding parliamentary speeches to trace MP/party trajectories over time, validated against political scores.

## Project Overview
This project uses encoder-only language models to build vector representations of parliamentary speeches and the MPs who give them, then studies how those representations evolve over time to trace ideological trajectories. Because an embedding space is only useful if its geometry is politically meaningful, the focus is validation: testing whether the representation recover known structure such as party membership, survey ideology scores and real political events.

## Data
- Open Discourse Dataset
- CHES / Manifesto Project

## Repo Structure
- /data
- /notebooks
