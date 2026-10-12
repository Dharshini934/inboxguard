# InboxGuard - Spam / Ham Message Classifier

Machine-learning spam filter with a Flask website. It classifies emails/SMS as spam or ham, explains the decision word by word, and can scan a whole CSV.

## Run
```bash
pip install -r requirements.txt
python app.py            # open http://127.0.0.1:5000
python train.py          # optional: re-train, regenerate metrics and charts
```

## How it works
1. **Data** - SMS Spam Collection (UCI / Kaggle). 5,169 unique messages after removing 403 duplicates (653 spam, 4,516 ham). Any CSV whose first two columns are *label, text* can replace `data/spam.csv`.
2. **Preprocessing** (`inboxguard/preprocess.py`) - lowercase, links/emails/amounts/phone numbers/numbers replaced by special tokens, punctuation removed.
3. **Features** (`inboxguard/features.py`) - word TF-IDF (1-2 grams) + character TF-IDF (3-5 grams) + 11 style statistics (capital ratio, digit ratio, exclamations, has link/amount/phone, ...).
4. **Models** (`inboxguard/models.py`) - Naive Bayes, Logistic Regression, SVM.
5. **Low false positives** - the decision threshold of every model is tuned with 5-fold cross-validation on the training set so that at least 99% of flagged messages are truly spam. The final model is chosen by cross-validated average precision, never by the test set.
6. **Evaluation** - stratified 75/25 split, 1,293 unseen test messages.

## Results (test set, tuned threshold)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Ham flagged as spam | Spam missed |
|---|---|---|---|---|---|---|---|
| Naive Bayes | 98.1% | 97.3% | 87.1% | 0.919 | 0.979 | 4 | 21 |
| Logistic Regression | 98.3% | 100.0% | 86.5% | 0.928 | 0.992 | 0 | 22 |
| SVM (final) | 98.3% | 99.3% | 87.1% | 0.928 | 0.991 | 1 | 21 |

Higher precision means fewer legitimate messages end up in the spam folder; the trade-off is that about 13% of spam is let through.

## Website
- **Scan a message** - verdict stamp, spam score meter, words that pushed the decision, message facts
- **Model results** - dataset facts, model table, six charts
- **Scan a CSV** - upload a file with a `text` column, download the results

## Folder map
```
app.py                 Flask server and API
train.py               training, threshold tuning, charts
inboxguard/            preprocessing, features, models, explanations
templates/ static/     website
data/ models/ reports/ dataset, saved model, charts + metrics
```
