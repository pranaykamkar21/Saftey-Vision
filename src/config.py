from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

MASK_CLASSES = ["with_mask", "without_mask", "mask_worn_incorrectly"]
HELMET_CLASSES = ["helmet", "no_helmet"]

DEFAULT_CONFIDENCE = 0.5
DEFAULT_IMG_SIZE = 640
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 50
