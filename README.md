# Consumer Demand Model Explorer

A first-draft Streamlit application that solves and plots the two-good consumer problem for:

- Cobb–Douglas preferences
- addictive/non-convex preferences, `u(x1, x2) = x1^2 + x2^2`
- neutral goods
- perfect substitutes
- perfect complements
- quasi-linear preferences

The app has two study modes:

- **Explore one model** shows its optimal bundle, explanation, and graph in detail.
- **Compare all models** holds prices and income constant, displays six graphs side by side,
  and summarizes how each preference structure changes the optimal choice.

## Run it

Use Python 3.10 or newer. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Run the automated checks with:

```bash
pytest -q
```

## Current limitations

This draft uses fixed functional forms, two goods, positive prices, and analytical solutions. A later version could add demand curves over changing prices, numerical optimization, downloadable results, and more quasi-linear subtypes.
