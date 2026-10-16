# Disease screening and modeling — Arthur Costa

Five interactive projects with dedicated URLs, built with Streamlit.

[Projects portfolio](https://www.costa-risk-modeling.org/)

| Project | Live demo |
| --- | --- |
| Chronic disease screening | https://www.costa-risk-modeling.org/chronic-disease |
| Prostate cancer screening | https://www.costa-risk-modeling.org/prostate-cancer |
| Individual sensitivity analysis | https://www.costa-risk-modeling.org/sensitivity |
| Population scenario modeling | https://www.costa-risk-modeling.org/population |
| Model training and validation | https://www.costa-risk-modeling.org/validation |

[Read the foundational study](https://doi.org/10.64898/2025.12.10.25341981).

## Run locally

Use Python 3.12 and the package versions in `requirements.txt`, which match the saved model artifacts.

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On macOS, XGBoost also requires the OpenMP runtime (`libomp`).

Keep the model and data files alongside `app.py`. The app resolves their paths relative to its own directory. Optional Google Sheets logging uses `GOOGLE_SERVICE_ACCOUNT_JSON`; it is skipped when credentials are absent. Do not commit credentials.

## Navigation and inputs

Each project renders on its own route. The sensitivity project uses the baseline profile calculated in the chronic-disease project during the same session. The older `?page=...` links remain supported, including the embedded Oxum screener.

Height, weight, age, and sex are required for chronic-disease screening. Empty required fields display “Required” in red, and submission stays disabled until they are completed. Optional profile inputs start with “Don't know/Prefer not to say.” Chronic-disease predictions use the fitted model pipelines to handle skipped optional inputs. Age also displays “Required” in the prostate calculator. Its fixed formula requires all three inputs before calculating.
