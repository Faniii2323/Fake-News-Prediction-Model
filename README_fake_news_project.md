# Fake News Prediction - VS Code Project

## Dataset detected
The uploaded `archive.zip` contains:

- `News _dataset/Fake.csv`
- `News _dataset/True.csv`

Both files have these columns:

- `title`
- `text`
- `subject`
- `date`

The script automatically adds labels:

- Fake News = 1
- Real News = 0

## How to run

1. Create a folder in VS Code.
2. Put these files in that folder:
   - `fake_news_full_pipeline.py`
   - `requirements.txt`
   - your `archive.zip`

3. Open VS Code terminal and run:

```bash
python -m venv venv
```

Windows PowerShell:

```bash
.\venv\Scripts\Activate.ps1
```

Install libraries:

```bash
pip install -r requirements.txt
```

Run:

```bash
python fake_news_full_pipeline.py
```

## Output

The script will create an `outputs/` folder containing:

- `results_summary.csv`
- accuracy comparison graph
- F1-score comparison graph
- all metrics comparison graph
- confusion matrix for every model
- ROC curve for every model
- accuracy/loss plots for CNN, RNN, BI-LSTM, and GNN

## Models included

1. Logistic Regression
2. SVM
3. KNN
4. Random Forest
5. Decision Tree
6. Boosting
7. CNN
8. RNN
9. BI-LSTM
10. GNN
