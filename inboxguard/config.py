from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "spam.csv"
MODEL_PATH = ROOT / "models" / "inboxguard_model.joblib"
EXPLAINER_PATH = ROOT / "models" / "inboxguard_explainer.joblib"
REPORTS = ROOT / "reports"

SEED = 7
TEST_SIZE = 0.25
MIN_PRECISION = 0.99          # target: at most ~1% of flagged messages are really ham
