# Lead Score API

A real trained classifier that scores inbound leads by conversion likelihood — not an LLM prompt wrapper, an actual model trained on real marketing data with measured accuracy.

**Live:** https://lead-score-api-three.vercel.app — try `/docs` for the interactive API explorer, or `POST /score` directly.

## Why a trained model, not an LLM call

Calling an AI API per-lead (like the [n8n AI Lead Auto-Responder](https://github.com/ansarrahim/n8n-templates)) works well for unstructured tasks like drafting a reply. Scoring is a repeated, structured decision — train once, score thousands of leads in milliseconds, no per-request API cost. This project pairs naturally with that workflow: score a lead here first, then only trigger the (paid, slower) AI-drafted reply for the leads worth replying to personally.

## The data

Trained on a public, real-world dataset of ~9,200 leads from an online education company's marketing campaigns — [source](https://github.com/drajesh-tech/Logistic-Regression-Lead-Scoring-Case-Study), a well-known real lead-scoring case study dataset. Columns filled in by sales reps *after* contact (Tags, Lead Quality, Last Notable Activity) are deliberately excluded from training to avoid leakage — the model only sees signals genuinely available at the moment a lead comes in: origin, source, engagement (visits, time on site, page views), last recorded activity, and occupation/specialization.

**This ships trained on public benchmark data as a working demo.** For real production use, retrain `train.py` on your own historical lead data — the pipeline is built to make that a one-line swap of `data/Leads.csv`.

## Model performance

Gradient boosting classifier, evaluated on a held-out 20% test split (1,848 leads):

| Metric | Value |
| --- | --- |
| Accuracy | 82.2% |
| Precision | 76.9% |
| Recall | 77.0% |
| F1 | 76.9% |
| ROC-AUC | 0.888 |

A logistic regression baseline was also trained for comparison (80.1% accuracy, 0.875 ROC-AUC) — gradient boosting won out and is what's deployed.

Run `python train.py` to reproduce — numbers get written fresh to `model/metrics.json` each time, not hand-picked.

## API

```
POST /score
{
  "lead_origin": "Landing Page Submission",
  "lead_source": "Google",
  "do_not_email": "No",
  "total_visits": 5,
  "total_time_spent_on_website": 720,
  "page_views_per_visit": 2.5,
  "last_activity": "Email Opened",
  "specialization": "Business Administration",
  "current_occupation": "Working Professional"
}
```

Returns:

```
{ "conversion_probability": 0.71, "score_band": "Hot" }
```

`GET /health` reports whether the model is loaded.

## Local development

```bash
python -m venv .venv
.venv/Scripts/activate   # or source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
python train.py          # trains the model, writes model/lead_score_model.joblib
uvicorn app.main:app --reload
```

## Stack

Python, pandas, scikit-learn (GradientBoostingClassifier, with a logistic regression baseline for comparison), FastAPI, deployed on Render.
