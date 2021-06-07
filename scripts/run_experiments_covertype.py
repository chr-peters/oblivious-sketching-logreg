import logging

from sketching import settings
from sketching.datasets import Covertype_Sklearn
from sketching.utils import run_experiments

logger = logging.getLogger(settings.LOGGER_NAME)


MIN_SIZE = 500
MAX_SIZE = 15000
STEP_SIZE = 500
NUM_RUNS = 21

dataset = Covertype_Sklearn()

run_experiments(
    dataset=dataset,
    min_size=MIN_SIZE,
    max_size=MAX_SIZE,
    step_size=STEP_SIZE,
    num_runs=NUM_RUNS,
)
