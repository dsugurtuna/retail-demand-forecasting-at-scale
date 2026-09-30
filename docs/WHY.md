# Why it's built this way

## The problem

A retailer has to decide every week how much of each product to send to each store, and both running out and over-stocking cost money. The job is to forecast daily unit sales for thousands of item-store series 28 days ahead, and to prove the forecasts are better than simple rules before anyone relies on them.

## Design choices

**Why one LightGBM model for all series, rather than one model per series?**
Because most retail series are short, noisy and full of zero days, so a model per series has little to learn from. One global model learns shared patterns (weekends, promotions, holidays) from every series at once, handles numeric and categorical inputs together, trains in minutes on a CPU, and is one thing to maintain rather than thousands.

**Why is every target-based feature at least 28 days old (`min_lag` = horizon)?**
Because it lets one model forecast every day of the 28-day horizon directly from what is known at the forecast origin. The alternative, recursive forecasting, feeds the model's own predictions back in as inputs, so errors compound and the training setup no longer matches how the model is used. The cost is real: the forecast for tomorrow ignores the last 27 days of sales. See "Questions worth asking" below.

**Why hide the target after the origin as well as shifting features?**
Because the shift is a rule, and rules get broken by the next feature someone adds. Setting future target values to null before any feature is built means a wrong feature reads nulls, not answers. A test scrambles those future values and checks that no feature changes, and the smoke run repeats that check on every run.

**Why a Tweedie objective by default?**
Because unit sales are non-negative counts with many zeros and a long tail. The Tweedie family sits between Poisson and gamma and fits that shape; plain squared error treats a miss of 2 units on a slow mover the same as on a best-seller and can predict negative sales.

**Why WRMSSE, and why at 12 levels?**
Because it is the metric the M5 competition used for exactly this data shape. Scaling each series by its own day-to-day volatility makes slow and fast movers comparable. Weighting by recent dollar sales makes the score care about what matters to revenue. Scoring at store, department and total levels as well as item-store rewards forecasts that add up sensibly where stock decisions are actually made.

**Why score against naive baselines in every run?**
Because a WRMSSE score means nothing on its own. "Repeat last week" and "the last 28-day average" are what a planner could do with a spreadsheet. If the model cannot beat them, it is not earning its complexity, and the smoke run fails.

**Why does the scheduled job train on synthetic data?**
Because a weekly check must be free, fast, offline and repeatable. The old job ran for up to 80 minutes a week, tried to install PyTorch and a dozen libraries the code never imports, pointed at a training module that did not exist, and failed every time without testing anything. The synthetic generator has known structure (weekday, season, events, promotions), so "the model beats naive baselines" is a meaningful pass/fail test of the pipeline. Training on the real M5 data needs a Kaggle account and more memory than a hosted runner has, so it is a deliberate manual run.

**Why keep missing values instead of filling them with 0?**
Because 0 is a claim ("sold nothing") and missing is the truth ("we do not know yet"). LightGBM learns which way to send missing values at each split, so nothing is gained by inventing a number.

**Why does validation block training?**
Because fixing the data comes before modelling. A model trained on duplicated days or negative sales still produces confident numbers; they are just wrong. Pandera checks types and ranges, plain rules check duplicates and history length, and optional columns such as event names only warn.

**Why write metrics to JSON rather than an experiment tracker?**
Because the run has to work offline with no services, in CI and on a laptop. Choosing and operating a tracking server is a separate decision from building the model; the JSON artefacts can be logged to any tracker later without changing the pipeline.

**Why is the API described as a skeleton?**
Because the model needs recent sales history for each series, and the API does not look that up yet. It would be easy to return plausible-looking numbers; saying clearly that they are not the scored forecasts is more useful than hiding the gap.

**Why deterministic training?**
Because every number in the README must be reproducible by a command. Fixed seeds and LightGBM's deterministic mode mean the same data, settings and platform give the same model, and the smoke run checks that a saved model reloads with identical forecasts.

## Questions worth asking

**"Your model beats naive baselines on data you generated yourself. Isn't that circular?"**
Partly, and that is why the result is labelled as a pipeline check, not an accuracy claim. The generator decides what signal exists; the check asks whether the pipeline can find it. That still catches the failures that matter operationally: leakage turned off, misaligned rows, a broken model, a library upgrade that changes behaviour. A pipeline with any of those bugs loses to "repeat last week". Accuracy on real data needs the M5 run (`python -m src.train --data-dir data/raw --stores CA_1`), and no M5 result is published here because none has been produced by this code yet.

**"How do you know there is no leakage?"**
Four layers: features are shifted by `min_lag`; future targets are hidden before features are built; early stopping uses dates before the origin; and tests scramble future values and check nothing changes. The backtest's last fold uses the same origin as the holdout and reproduces its score exactly, which it would not do if the two paths saw different data. What remains: prices are treated as known through the horizon, as in M5, and two price features use each series' whole price history within the frame. In a real business, promotional plans change, so those assumptions should be checked with the people who set prices.

**"Using 28-day-old data for tomorrow's forecast throws information away. Why accept that?"**
It does, and for near-term days it probably costs accuracy. The trade-off buys a single model, no compounding errors, and a training setup that matches use. The standard fix is a small set of models by horizon (for example days 1-7, 8-14, 15-28), each with its own `min_lag`. The backtest engine already reports error by horizon day, which is the measurement that comparison needs; the comparison itself has not been run, so there is no claim either way.

## What comes next

1. Run the pipeline on the real M5 data for one store and publish the command, the scores and the baselines, including where the model loses.
2. Compare horizon-bucketed models against the single 28-day model in rolling backtests.
3. Compare the WRMSSE surrogate objective with Tweedie on the same folds before recommending either.
4. Replace the ±20% placeholder band with quantile or conformal intervals, and report their measured coverage.
5. Give the API a history lookup so that it serves the same forecasts the training run scores.
6. Build the Docker images in CI and read the YAML settings from the training command.
