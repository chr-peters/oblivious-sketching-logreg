from setuptools import find_packages, setup

setup(
    name="oblivious-sketching-logreg",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["joblib", "numba", "pandas", "scikit-learn", "scipy"],
)
