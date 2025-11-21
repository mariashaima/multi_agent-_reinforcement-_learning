from setuptools import find_packages, setup

setup(
    name="marl_traffic_gen",
    packages=find_packages(where="src"),  # Includes all Python files
    package_dir={"": "src"},  # Tell distutils packages are under src
)
