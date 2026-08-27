# Tracking Political Change with Embeddings of Parliamentary Speech

Embedding German Bundestag speeches (2000–2021) and testing what those representations correspond to politically.

## Project Overview

We embed speeches from the Open Discourse Bundestag corpus twice, once with the off-the-shelf Jina v3 model and once with a copy fine-tuned contrastively on speaker identity, and aggregate both to the MP-year and party-year level. Since the embeddings cannot be inspected directly, we ask whether they recover structure that is known from outside the model: party membership, the Chapel Hill Expert Survey left-right scale, party movement around four political crises and the positions of individual MPs who change party. Alongside this we derive the topic structure of the debates (BERTopic and a zero-shot assignment to 21 policy fields) in order to separate subject matter from position.

## Repository Structure

```
notebooks/
├── 01_data_cleaning.ipynb                        cleaning, filtering, exploratory analysis
├── 02_speech_corpus_split.ipynb                  fine-tune, probe and embedding splits
├── 03_embedding_baseline.ipynb                   off-the-shelf Jina v3 embeddings
├── 04_finetuning_embedding.ipynb                 LoRA contrastive fine-tuning + party probe
├── 05_topic_embeddings.ipynb                     zero-shot topic embeddings
├── 06_aggregate_embeddings.ipynb                 MP-year and party-year aggregation
├── 07_test_embeddings.ipynb                      validation against party labels and CHES
├── 08_topic_analysis.ipynb                       BERTopic topic structure
├── 08b_topic__analysis_zeroshot.ipynb            same, zero-shot assignment
├── 09_party_movement_and_crisis.ipynb            party trajectories around the four events
├── 09b_party_movement_and_crisis_zeroshot.ipynb  same, zero-shot assignment
├── 10_party_switchers.ipynb                      MP-level trajectories across party switches
├── aggregation_functions.py                      aggregation helpers
└── embedding_utils.py                            mean_pooling, embed_speeches
```

Note: Notebooks 03–05 import `mean_pooling` and `embed_speeches` from `embedding_utils.py`, so this file has to sit next to the notebooks.

## Data

No data is stored in this repository. All datasets, the fine-tuned model, the embeddings and the topic assignments are in the university cloud:

https://cloud.uni-konstanz.de/index.php/f/248607213

The notebooks load files by plain filename, without a folder prefix. Download what a notebook needs and put it in the same folder as the notebooks, the cloud folders are only for organisation and are not mirrored in the code.

```
01_data/
├── raw/
│   ├── speeches.csv				        open discourse dataset
│   ├── CHES.csv				            chapel hill expert survey dataset
│   └── factions.csv				        auxiliary table
└── preprocessed/
    ├── speeches_main_2000_2021.csv	  	    cleaned speeches
    ├── speeches_main_2000_2021_trimmed.csv	same, opening formulas trimmed
    └── ches_de.csv				            filtered expert survey dataset

02_split_speech_corpus/
├── embedding_corpus.csv                  	82,447 speeches, main analysis
├── finetune_subsample.csv                	22,513 speeches, contrastive training
├── probe_val_sample.csv                  	720 speeches, checkpoint selection
└── probe_test_sample.csv                 	480 speeches, held-out comparison

03_model/
└── jina_v3_contrastive_backbone/         	fine-tuned encoder

04_embeddings/
├── raw/
│   ├── jina_v3_offtheshelf_full.parquet  	speech vectors, baseline
│   ├── jina_v3_contrastive_full.parquet  	speech vectors, fine-tuned
│   └── zeroshot_topic_embeddings.parquet 	vectors of the 21 policy fields
└── aggregated/
    ├── mp_year_embeddings_offtheshelf.parquet
    ├── mp_year_embeddings_finetuned.parquet
    ├── party_year_embeddings_offtheshelf.parquet
    └── party_year_embeddings_finetuned.parquet

05_topic_assignments/
├── speech_topic_assignments.csv           	BERTopic
└── speech_topic_assignments_zeroshot.csv  	zero-shot
```
