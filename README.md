# Mixed Doubles Curling Risk/Reward

## Quick start (required order)
1. `-r pip install requirements`
2. Import the data challenge CSVs into `data/`:
   - `data/Competition.csv`
   - `data/Competitors.csv`
   - `data/Games.csv`
   - `data/Teams.csv`
   - `data/Ends.csv`
   - `data/Stones.csv`
3. Run the two Python scripts (in this order):
   - `python code/stone_level_cleaning.py`
   - `python code/data_merger.py`

## File-by-file overview

### code/
- `code/stone_level_cleaning.py` — Builds a cleaned, stone-level master table by merging Stones/Ends/Games, infers hammer, computes pre-end scores, and writes `data/stones_master_1.csv`.
- `code/data_merger.py` — Builds an end-level dataset with hammer/non-hammer context, task counts, and team metadata; writes `data/end_level.csv`.

### notebooks/
- `notebooks/project.qmd` — Original class project notebook with data cleaning, EDA, hammer inference, and modeling (logistic regression/RF/XGBoost).
- `notebooks/eda.qmd` — End-level EDA on team strength, power play context, and logistic regression for hammer outcomes in PP ends.
- `notebooks/powerplay_exploration.qmd` — Focused PP analysis: usage timing, score context, outcomes, and shot-mix comparisons.
- `notebooks/stone_observation.qmd` — Stone-level exploratory plots; saves many figures to `notebooks/figures/stone_observation/`.
- `notebooks/fs_clustering.qmd` — Power play first-shot clustering (HDBSCAN), ANOVA/bootstrap CIs, and strategy grouping.
- `notebooks/behavior_regression.qmd` — Late-phase shot-selection behavior labels and regression analysis on power play effects.
- `notebooks/behavior_scoring.qmd` — Unused attempt at end “behavior scoring” based on board state changes.
- `notebooks/new_data_dictionary.qmd` — Revised data dictionary for the stone-level master table.

### writing/
- `writing/manuscript.tex` — Main paper draft (LaTeX) on power play strategy in mixed doubles curling.
- `writing/Makefile` — LaTeX build helpers (latexmk, diff, clean).
- `writing/references.bib` — BibTeX references for the manuscript.

### writing/tables/
- `writing/tables/cluster_test.tex` — Cluster means + ANOVA summary table for clustering results.
- `writing/tables/sensitivity_mapping_means.tex` — Sensitivity analysis means by task mapping.
- `writing/tables/sensitivity_mapping_reg.tex` — Sensitivity regression coefficients by task mapping.
- `writing/tables/team_strategy_profiles.tex` — Team strategy profile summary table.
- `writing/tables/regression_powerplay.tex` — Main PP regression table for behavior index.
