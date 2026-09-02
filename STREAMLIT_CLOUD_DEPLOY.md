# AirEquity Streamlit Community Cloud deployment

Entrypoint: `src/dashboard/app.py`

The app downloads the project's `v0.1-data/observations.zip` GitHub Release
asset on first run, retains the 2023 and 2024 partitions, and runs
`python -m src.features.build_features` if the feature table is missing.

Put the accompanying `requirements.txt` in `src/dashboard/`.

Deploy with:
- Repository: `ahriyad1/airequity-nsw`
- Branch: `main`
- Main file path: `src/dashboard/app.py`
