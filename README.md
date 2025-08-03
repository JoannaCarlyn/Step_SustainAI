## Dataset Overview

This project uses multiple dataset stages organized under the `data/` folder:

- `data/raw/`: (Optional) For storing real-world, unprocessed datasets (if collected in future).
- `data/synthetic/`: Manually created or programmatically generated dataset containing sample prompts, hardware type, and estimated energy consumption.
- `data/processed/`: Cleaned and enriched data including token count, readability score, etc., saved after applying preprocessing in the notebook.

The synthetic dataset serves as a stand-in for real-world energy and NLP data. All analysis and model training are based on this processed data.
