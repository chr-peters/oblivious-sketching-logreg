from sketching.datasets import Webspam_libsvm
from sketching.utils import run_experiments

MIN_SIZE = 500
MAX_SIZE = 15000
STEP_SIZE = 500
NUM_RUNS = 21

dataset = Webspam_libsvm()

run_experiments(
    dataset=dataset,
    min_size=MIN_SIZE,
    max_size=MAX_SIZE,
    step_size=STEP_SIZE,
    num_runs=NUM_RUNS,
)
