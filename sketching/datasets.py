import abc
import bz2
import io
import logging
import ssl
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype, fetch_kddcup99, load_svmlight_file
from sklearn.preprocessing import scale

from . import optimizer, settings

logger = logging.getLogger(settings.LOGGER_NAME)

_rng = np.random.default_rng()


def make_Z(X, y):
    # add intercept to X
    X_intercept = np.append(X, np.ones(shape=(X.shape[0], 1)), axis=1)

    # multiply row-wise by y
    Z = np.multiply(X_intercept, y[:, np.newaxis])

    return Z


class Dataset(abc.ABC):
    def __init__(self, use_caching, cache_dir=None):
        self.use_caching = use_caching
        if cache_dir is None:
            cache_dir = settings.DATA_DIR
        self.cache_dir = cache_dir

        if use_caching and not self.cache_dir.exists():
            self.cache_dir.mkdir()

        self.X = None
        self.y = None
        self.beta_opt = None

    @abc.abstractmethod
    def load_X_y(self):
        pass

    @abc.abstractmethod
    def get_name(self):
        pass

    def _load_X_y_cached(self):
        if not self.use_caching:
            logger.info("Loading X and y...")
            X, y = self.load_X_y()
            logger.info("Done.")
            return X, y

        X_path = self.get_binary_path_X()
        y_path = self.get_binary_path_y()
        if X_path.exists() and y_path.exists():
            logger.info(
                f"Loading cached versions of X and y found at {X_path} and {y_path}..."
            )
            X = np.load(X_path)
            y = np.load(y_path)
            logger.info("Done.")
            return X, y

        logger.info("Loading X and y...")
        X, y = self.load_X_y()
        logger.info("Done.")
        np.save(X_path, X)
        np.save(y_path, y)
        logger.info(f"Saved X and y at {X_path} and {y_path}.")

        return X, y

    def _get_beta_opt_cached(self):
        if not self.use_caching:
            Z = self.get_Z()
            logger.info("Computing beta_opt...")
            beta_opt = optimizer.optimize(Z).x
            logger.info("Done.")
            return beta_opt

        beta_opt_path = self.get_binary_path_beta_opt()
        if beta_opt_path.exists():
            logger.info(
                f"Loading cached version of beta_opt found at {beta_opt_path}..."
            )
            beta_opt = np.load(beta_opt_path)
            logger.info("Done.")
            return beta_opt

        Z = self.get_Z()
        logger.info("Computing beta_opt...")
        beta_opt = optimizer.optimize(Z).x
        logger.info("Done.")
        np.save(beta_opt_path, beta_opt)
        logger.info(f"Saved beta_opt at {beta_opt_path}.")

        return beta_opt

    def _assert_data_loaded(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_X_y_cached()

    def get_binary_path_X(self) -> Path:
        return self.cache_dir / f"{self.get_name()}_X.npy"

    def get_binary_path_y(self) -> Path:
        return self.cache_dir / f"{self.get_name()}_y.npy"

    def get_binary_path_beta_opt(self) -> Path:
        return self.cache_dir / f"{self.get_name()}_beta_opt.npy"

    def get_X(self):
        self._assert_data_loaded()
        return self.X

    def get_y(self):
        self._assert_data_loaded()
        return self.y

    def get_Z(self):
        self._assert_data_loaded()
        return make_Z(self.X, self.y)

    def get_n(self):
        self._assert_data_loaded()
        return self.X.shape[0]

    def get_d(self):
        self._assert_data_loaded()
        return self.X.shape[1]

    def get_beta_opt(self):
        if self.beta_opt is None:
            self.beta_opt = self._get_beta_opt_cached()

        return self.beta_opt


class NoisyDataset(Dataset):
    """
    This is a decorator class that adds gaussian noise to a subset of the rows
    of an existing dataset.

    Parameters
    ----------
    dataset : Dataset
        The dataset instance that will be augmented with noise.

    percentage : float
        Percentage of the dataset that will be augmented.
        Setting percentage=0.1 means that 10% of the rows in the dataset will be
        affected.

    std : float
        Standard deviation of the noise.
    """

    def __init__(self, dataset: Dataset, percentage, std):
        super().__init__(use_caching=False)
        self.dataset = dataset
        self.percentage = percentage
        self.std = std

    def get_name(self):
        return self.dataset.get_name() + "_noisy"

    def load_X_y(self):
        X, y = self.dataset.get_X(), self.dataset.get_y()

        subset_size = int(X.shape[0] * self.percentage)
        indices = _rng.choice(X.shape[0], size=subset_size, replace=False)
        X[indices] += _rng.normal(loc=0, scale=self.std, size=(subset_size, X.shape[1]))

        return X, y


class Covertype_Sklearn(Dataset):
    """
    Dataset Homepage:
    https://archive.ics.uci.edu/ml/datasets/Covertype
    """

    features_continuous = [
        "Elevation",
        "Aspect",
        "Slope",
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Hillshade_9am",
        "Hillshade_Noon",
        "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points",
    ]

    def __init__(self, use_caching=True):
        super().__init__(use_caching=use_caching)

    def get_name(self):
        return "covertype_sklearn"

    def load_X_y(self):
        logger.info("Fetching covertype from sklearn...")
        sklearn_result = fetch_covtype(as_frame=True)
        df = sklearn_result.frame

        logger.info("Preprocessing...")

        # Cover_Type 2 gets the label 1, everything else gets the label -1.
        # This ensures maximum balance.
        y = df["Cover_Type"].apply(lambda x: 1 if x == 2 else -1).to_numpy()
        df = df.drop("Cover_Type", axis="columns")

        # scale the continuous features to mean zearo and variance 1
        # and leave the 0/1 features as is
        X_continuous = df[self.features_continuous].to_numpy()
        X_continuous = scale(X_continuous)

        features_binary = list(set(df.columns) - set(self.features_continuous))
        X_binary = df[features_binary].to_numpy()

        # put binary features and scaled features back together
        X = np.append(X_continuous, X_binary, axis=1)

        return X, y


class KDDCup_Sklearn(Dataset):
    """
    Dataset Homepage:
    https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
    """

    features_continuous = [
        "duration",
        "src_bytes",
        "dst_bytes",
        "wrong_fragment",
        "urgent",
        "hot",
        "num_failed_logins",
        "num_compromised",
        "root_shell",
        "su_attempted",
        "num_root",
        "num_file_creations",
        "num_shells",
        "num_access_files",
        "num_outbound_cmds",
        "count",
        "srv_count",
        "serror_rate",
        "srv_serror_rate",
        "rerror_rate",
        "srv_rerror_rate",
        "same_srv_rate",
        "diff_srv_rate",
        "srv_diff_host_rate",
        "dst_host_count",
        "dst_host_srv_count",
        "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate",
        "dst_host_srv_serror_rate",
        "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate",
    ]

    features_discrete = [
        "protocol_type",
        "service",
        "flag",
        "land",
        "logged_in",
        "is_host_login",
        "is_guest_login",
    ]

    def __init__(self, use_caching=True):
        super().__init__(use_caching=use_caching)

    def get_name(self):
        return "kddcup_sklearn"

    def load_X_y(self):
        logger.info("Fetching kddcup from sklearn...")
        sklearn_result = fetch_kddcup99(as_frame=True, percent10=True)
        df = sklearn_result.frame

        logger.info("Preprocessing...")

        # convert label "normal." to -1 and everything else to 1
        y = df.labels.apply(lambda x: -1 if x.decode() == "normal." else 1).to_numpy()

        # get all the continuous features
        X_continuous = df[self.features_continuous]

        # the feature num_outbound_cmds has only one value that doesn't
        # change, so drop it
        X_continuous = X_continuous.drop("num_outbound_cmds", axis="columns")

        # convert to numpy array
        X_continuous = X_continuous.to_numpy()

        # scale the features to mean 0 and variance 1
        X = scale(X_continuous)

        return X, y


class Webspam_libsvm(Dataset):
    """
    Dataset Source:
    https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#webspam
    """

    dataset_url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/webspam_wc_normalized_unigram.svm.bz2"  # noqa: E501

    def __init__(self, drop_sparse_columns=True, use_caching=True):
        self.drop_sparse_columns = drop_sparse_columns
        super().__init__(use_caching=use_caching)

    def get_name(self):
        if self.drop_sparse_columns:
            return "webspam_libsvm_desparsed"
        else:
            return "webspam_libsvm"

    def get_raw_path(self):
        return self.cache_dir / f"{self.get_name()}.csv"

    def download_dataset(self):
        logger.info(f"Downloading data from {self.dataset_url}")

        context = ssl._create_unverified_context()
        with urllib.request.urlopen(self.dataset_url, context=context) as f:
            contents = f.read()

        logger.info("Download completed.")
        logger.info("Extracting data...")

        file_raw = bz2.open(io.BytesIO(contents))
        X_sparse, y = load_svmlight_file(file_raw)

        # convert scipy Compressed Sparse Row array into numpy array
        X = X_sparse.toarray()

        df = pd.DataFrame(X)
        df["LABEL"] = y

        logger.info(f"Writing .csv file to {self.get_raw_path()}")

        df.to_csv(self.get_raw_path(), index=False)

    def load_X_y(self):
        if not self.get_raw_path().exists():
            logger.info(f"Couldn't find dataset at location {self.get_raw_path()}")
            self.download_dataset()

        df = pd.read_csv(self.get_raw_path())

        logger.info("Preprocessing the data...")

        y = df["LABEL"].to_numpy()
        df = df.drop("LABEL", axis="columns")

        # drop all columns that only have constant values
        # drop all columns that contain only one non-zero entry
        for cur_column_name in df.columns:
            cur_column = df[cur_column_name]
            cur_column_sum = cur_column.astype(bool).sum()
            unique_values = cur_column.unique()
            if len(unique_values) <= 1:
                df = df.drop(cur_column_name, axis="columns")
            if self.drop_sparse_columns and cur_column_sum == 1:
                df = df.drop(cur_column_name, axis="columns")

        X = df.to_numpy()

        # scale the features to mean 0 and variance 1
        X = scale(X)

        return X, y


class Synthetic_Dataset(Dataset):
    def __init__(self, n_rows, use_caching=False):
        self.n_rows = n_rows
        super().__init__(use_caching=use_caching)

    def get_name(self):
        return f"synthetic_n_{self.n_rows}"

    def load_X_y(self):
        pos_label = np.array([-2, -1, 1]).reshape((1, 3))
        neg_label = np.array([-2, 1, -1]).reshape((1, 3))
        outlier_1 = np.array([0, self.n_rows, 1]).reshape((1, 3))
        outlier_2 = np.array([0, self.n_rows, -1]).reshape((1, 3))

        pos_label_block = np.repeat(pos_label, int(self.n_rows / 2), axis=0)
        neg_label_block = np.repeat(neg_label, int(self.n_rows / 2), axis=0)

        block = np.vstack([pos_label_block, neg_label_block, outlier_1, outlier_2])

        # shuffle
        block = np.random.permutation(block)

        X = block[:, :-1]
        y = block[:, -1]

        return X, y
