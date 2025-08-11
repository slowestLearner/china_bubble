# China Bubble Project

---

## Setup and Installation

1.  Ensure you have Python 3.13.5 (we use `pyenv`, .python-version, to control that) and `uv` installed.

2.  Create the virtual environment:
    ```bash
    uv venv
    ```
3.  Activate the environment:
    ```bash
    source .venv/bin/activate
    ```
4.  Install the required dependencies:
    ```bash
    uv pip sync pyproject.toml
    ```
---

## Scripts

This section details the data processing and analysis scripts. They should be run in the order listed.

### `1_stock_data_processing.py`

Computes stock characteristics and saves the processed data and a summary table.


### `2_descriptive_analysis.py`

Produces descriptive price path plots. 


### `3_boom_bust_stock_analysis.py`

- Create summary statistics about "boom bust stocks". 
- Identify stocks that experience large vs small bubbles
- Estimate and plot differences in characteristics between bubble vs non-bubble stocks

### `4_retail_entries.py`

- Plot new entry of retail investors
- Plot net retail flows (TODO: should we plot retail flows, and possibly also new entries, on bubble vs non-bubble stocks?)

### `5_top_coded_iv_construction.py`

- Estimate KY preference parameters of investors

### 





