# OrderSense-AI

OrderSense-AI is organized into three parts: return prediction, product classification, and an AI support agent.

## Project structure

- `dataset/`: shared source data
- `part1_return_prediction/`: return prediction data, notebooks, source, models, and app
- `part2_product_classification/`: product classification data, notebooks, source, models, and app
- `part3_ai_support_agent/`: support-agent source, prompts, and app
- `common/utils/`: shared utilities

## Getting started

Activate the existing virtual environment in PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Install project dependencies as they are added, then run the existing data loader:

```powershell
python load_data.py
```

Run the starter checks with:

```powershell
python -m unittest test.py
```
