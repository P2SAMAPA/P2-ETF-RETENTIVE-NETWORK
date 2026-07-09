# P2-ETF-RETENTIVE-NETWORK# Retentive Network (RetNet) for ETFs

Implements the Retentive Network (RetNet) – a successor to Transformers with O(1) inference complexity. Multi-scale retention replaces attention with a chunk-wise recurrent formulation. The model predicts next‑day ETF returns from sequences of ETF returns and macro variables.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Multi-scale retention with chunk-wise recurrence
- Configurable hidden size, heads, layers, sequence length
- Score = predicted next‑day return
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-retentive-network-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (slower due to neural net training)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High positive score → ETF expected to rise tomorrow.
- Negative score → expected to fall.

## Requirements

See `requirements.txt`.
