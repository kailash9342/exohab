---
description: Runnable exohab examples - a Jupyter notebook walkthrough, a script that tests every function, and an installation self-check.
---

# Examples

| Example | What it shows |
|---|---|
| [Quickstart notebook](https://github.com/kailash9342/exohab/blob/main/examples/quickstart.ipynb) | A Jupyter walkthrough with saved outputs: metrics, live NASA data, Monte Carlo for one planet and the whole catalogue, plots and ML features |
| [Full tour script](https://github.com/kailash9342/exohab/blob/main/examples/full_tour.py) | One script that runs and checks every public function, saves three plots, and ends with a pass/fail summary |
| [Installation self-check](https://github.com/kailash9342/exohab/blob/main/examples/try_exohab.py) | A short check of your installation against published values |

## Run the full tour

```bash
pip install "exohab[all]"
curl -o full_tour.py https://raw.githubusercontent.com/kailash9342/exohab/main/examples/full_tour.py
python full_tour.py
```

It should end with:

```text
[PASS] every public name exercised (28/28)

   26 passed, 0 failed, 0 skipped
```

## Open the notebook

```bash
pip install "exohab[all]" notebook
curl -o quickstart.ipynb https://raw.githubusercontent.com/kailash9342/exohab/main/examples/quickstart.ipynb
jupyter notebook quickstart.ipynb
```

You can also read the notebook with all its outputs directly on
[GitHub](https://github.com/kailash9342/exohab/blob/main/examples/quickstart.ipynb).
