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

This script computes key characteristics, and saves the processed data and a summary table.


### `2_descriptive_analysis.py`

This script takes the processed data and generates descriptive plots and tables for the analysis portion of the project.

