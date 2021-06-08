from sketching.datasets import NoisyDataset, Webspam_libsvm
from sketching.utils import run_experiments

MIN_SIZE = 500
MAX_SIZE = 15000
STEP_SIZE = 500
NUM_RUNS = 21

dataset_noisy = NoisyDataset(dataset=Webspam_libsvm(), percentage=0.01, std=10)

run_experiments(
    dataset=dataset_noisy,
    min_size=MIN_SIZE,
    max_size=MAX_SIZE,
    step_size=STEP_SIZE,
    num_runs=NUM_RUNS,
)
