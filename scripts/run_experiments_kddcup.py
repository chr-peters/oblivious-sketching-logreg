from sketching.datasets import KDDCup_Sklearn
from sketching.utils import run_experiments

MIN_SIZE = 1000
MAX_SIZE = 30000
STEP_SIZE = 1000
NUM_RUNS = 21

dataset = KDDCup_Sklearn()

run_experiments(
    dataset=dataset,
    min_size=MIN_SIZE,
    max_size=MAX_SIZE,
    step_size=STEP_SIZE,
    num_runs=NUM_RUNS,
)
