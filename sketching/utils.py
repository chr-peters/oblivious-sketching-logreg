import logging

from . import settings
from .datasets import Dataset
from .experiments import (
    L2SExperiment,
    ObliviousSketchingExperiment,
    SGDExperiment,
    UniformSamplingExperiment,
)

logger = logging.getLogger(settings.LOGGER_NAME)


def run_experiments(dataset: Dataset, min_size, max_size, step_size, num_runs):
    # check if results directory exists
    if not settings.RESULTS_DIR.exists():
        settings.RESULTS_DIR.mkdir()

    logger.info("Starting uniform sampling experiment")
    experiment_uniform = UniformSamplingExperiment(
        dataset=dataset,
        results_filename=settings.RESULTS_DIR / f"{dataset.get_name()}_uniform.csv",
        min_size=min_size,
        max_size=max_size,
        step_size=step_size,
        num_runs=num_runs,
    )
    experiment_uniform.run(parallel=True)

    logger.info("Starting L2S experiment")
    experiment_l2s = L2SExperiment(
        dataset=dataset,
        results_filename=settings.RESULTS_DIR / f"{dataset.get_name()}_l2s.csv",
        min_size=min_size,
        max_size=max_size,
        step_size=step_size,
        num_runs=num_runs,
    )
    experiment_l2s.run(parallel=True)

    logger.info("Starting sketching experiment")
    experiment_sketching = ObliviousSketchingExperiment(
        dataset=dataset,
        results_filename=settings.RESULTS_DIR / f"{dataset.get_name()}_sketching.csv",
        min_size=min_size,
        max_size=max_size,
        step_size=step_size,
        num_runs=num_runs,
        h_max=2,
        kyfan_percent=0.25,
    )
    experiment_sketching.run(parallel=False)

    logger.info("Starting SGD experiment")
    experiment_sgd = SGDExperiment(
        dataset=dataset,
        results_filename=settings.RESULTS_DIR / f"{dataset.get_name()}_sgd.csv",
        num_runs=num_runs,
    )
    experiment_sgd.run(parallel=False)
