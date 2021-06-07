from sketching.datasets import Synthetic_Dataset
from sketching.utils import run_experiments

MIN_SIZE = 100
MAX_SIZE = 3000
STEP_SIZE = 100
NUM_RUNS = 21

dataset = Synthetic_Dataset(n_rows=100000)

run_experiments(
    dataset=dataset,
    min_size=MIN_SIZE,
    max_size=MAX_SIZE,
    step_size=STEP_SIZE,
    num_runs=NUM_RUNS,
)
