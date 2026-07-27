# Dynamic Pricing of Diamonds

Modeling how diamond attributes drive price, and simulating a quality-based dynamic pricing strategy — comparing a classical econometric model against a neural network on the same data.

## What it does

1. **Feature Engineering** — cleans the raw diamonds dataset, computes volume (`x × y × z`), flags high-quality stones (Ideal cut + top clarity grades), and applies a 5% price premium to those stones to create a `dynamic_price` target.
2. **Statistical Analysis** — computes mean, standard deviation, and coefficient of variation to characterize price volatility across the dataset.
3. **Regression Modeling** — fits both a simple (carat-only) and multiple linear regression (statsmodels OLS) to explain dynamic price, checking significance and multicollinearity.
4. **Neural Network** — trains a scikit-learn `MLPRegressor` on the same features (scaled), as a non-linear comparison point against the OLS model, evaluated with R² and MSE on a held-out test set.
5. **Correlation Analysis** — a masked heatmap of all numeric features against dynamic price.
6. **Timeline Simulation** — assigns synthetic sequential timestamps to the (otherwise static) data to demonstrate time-series trend analysis on the simulated pricing strategy.
7. **Segment Analysis** — a boxplot of dynamic price distribution across diamond cut quality.

## Key findings

- The multiple regression explains ~85.7% of price variance (R² = 0.857); carat is the dominant driver (~$7,746 per carat added), with the high-quality flag contributing a smaller, independent ~$1,123.
- Carat and volume are near-perfectly correlated, which explains the multicollinearity flagged in the regression.
- The neural network offers a non-linear point of comparison against the OLS model on the same test data.

## Tech stack

`Python` · `Jupyter Lab` · `pandas` · `NumPy` · `statsmodels` · `scikit-learn` · `matplotlib` · `seaborn`

## Running it

```bash
pip install pandas numpy statsmodels matplotlib seaborn scikit-learn
```

Place `Diamonds_Prices2022.csv` in the same directory as the notebook and run top to bottom.

## Notes

- The 5% "dynamic pricing" premium and the simulated daily timestamps are both modeling techniques applied to a static dataset — they stand in for the kind of live, timestamped data a real dynamic-pricing system would use, and are labeled as such in the notebook.
