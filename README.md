# PsychoScape: Mapping the Pyschology Research Landscape

Fork of the NeuroScape repository.

This repository contains scripts and notebooks for analyzing and visualizing research in the domain of psychology through large-scale data collection, filtering, clustering, and semantic analysis of scientific articles.

## Repository Structure
```
.
├── config/                 # TOML config files (scraping, clustering, analysis, etc.)
├── notebooks/              # Jupyter notebooks for exploration and results
├── scripts/                # Main processing scripts organized by function
│   ├── ingestion/          # Data collection and cleaning
│   ├── preprocessing/      # Filtering and classification
│   ├── domain_embedding/   # Training & applying domain-specific embeddings
│   ├── clustering/         # Building semantic graphs & community detection
│   ├── graph_analysis/     # Citation/network density analysis
│   └── semantic_analysis/  # Dimension analysis, cluster characterization, trends, etc.
├── src/
│   ├── classes/            # Python classes for data structures and model architectures
│   └── utils/              # Utility modules (parsing, data loading, plotting, etc.)
└── README.md
```

## Data

This repository has been used to collect, curate, and analyze the following dataset:

Senden, M. (2025). NeuroScape (1.0.1) [Data set]. Zenodo. [https://doi.org/10.5281/zenodo.14865161](https://doi.org/10.5281/zenodo.14865161)

## Workflow Overview

1. **Scrape Data**
   - *scripts/ingestion/scraping.py*  
   - Query PubMed for relevant psychology articles.

3. **Merge and Clean**  
   - *scripts/ingestion/merge_and_clean.py*  
   - Consolidate scraped data, remove duplicates, and clean metadata.

4. **Initial Embedding**  
   - *scripts/ingestion/initial_embedding.py*  
   - Generate general-purpose text embeddings (via Voyage AI) for each abstract.

5. **Filter Data**  
   - *scripts/preprocessing/filter_disciplines.py*  
   - Retain only articles classified as psychology with high confidence. Use discipline classifier from NeuroScape.

6. **Train Domain Embedding Model**
    - *scripts/domain_embedding/train_embedding_model.py*
    - Trains a domain-specific embedding model on top of the initial embeddings (contrastive learning)

7. **Domain Embedding**
    - *scripts/domain_embedding/embed_abstracts.py*
    - Re-embeds abstracts in a lower-dimensional, psychology-focused space for semantic clustering.

8. **Build the Semantic Graph**
    - *scripts/clustering/graph_construction.py*
    - Uses the domain-specific embeddings to construct a similarity graph (e.g., KNN) needed for community detection.

9. **Community Detection**  
    - *scripts/clustering/community_detection.py*  
    - Perform clustering (e.g., Leiden community detection) on the network.

10. **Cluster Definition**  
    - *scripts/semantic_analysis/cluster_definition.py*  
    - Generate descriptive titles, keywords, and summaries for each cluster.

11. **Cluster Distinction**  
    - *scripts/semantic_analysis/cluster_distinction.py*  
    - Identify key differences between similar clusters.

12. **Dimensions Extraction**  
    - *scripts/semantic_analysis/assess_dimensions.py*  
    - Analyze each cluster across multiple research dimensions (e.g., appliedness, modality).

13. **Dimension Categorization**  
    - *scripts/semantic_analysis/assess_dimension_categories.py*  
    - Categorize clusters along specific sub-dimensions (e.g., spatial vs. temporal scales).

14. **Open Questions**  
    - *scripts/semantic_analysis/extract_open_questions.py*  
    - Identify important open research questions from recent review articles.

15. **Trends Extraction**  
    - *scripts/semantic_analysis/extract_trends.py*  
    - Compare older vs. recent publications to reveal emerging and declining trends.
