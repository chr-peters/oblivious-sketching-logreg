from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

# the downloaded datasets will go here
DATA_DIR = BASE_DIR / ".data-cache"
if not DATA_DIR.exists():
    DATA_DIR.mkdir()

RESULTS_DIR = BASE_DIR / "experimental-results"
if not RESULTS_DIR.exists():
    RESULTS_DIR.mkdir()

LOGGER_NAME = "sketching"
