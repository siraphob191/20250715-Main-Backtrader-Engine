# Using Jupyter Notebooks

This guide walks through setting up a small environment to run the example notebook that executes the backtest.

## 1. Clone the repository

```bash
git clone <repository-url>
cd 20250627-Momentum4-6-Refactor
```

## 2. Create a Conda environment

If you do not have Conda installed, install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) first. Then run:

```bash
conda create -n momentum46 python=3.10
conda activate momentum46
```

## 3. Install dependencies

Install the packages required by the project and the notebook interface. The project should be installed in editable mode so the `src` package can be imported without manually editing `sys.path`.

```bash
pip install -r requirements.txt
pip install jupyter
pip install -e .
```

## 4. Launch the notebook

Start Jupyter and open the example notebook in the `examples/` folder:

```bash
jupyter notebook
```

In the browser that appears, open `examples/RunBacktest.ipynb`. The first code cell imports `src.config` and defines the CSV paths—update those variables to point to your local data. Then run the remaining cells to execute the backtest and view the resulting summary tables.  No manual `sys.path` modification is required.
