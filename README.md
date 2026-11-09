# Give Me Some Credit — Credit Scoring Model

## Quickstart Usage

```python
import sys; sys.path.insert(0, "src")
from predict import score_applicant

r = score_applicant(dict(
    age=45,
    monthly_income=5000,
    debt_ratio=0.3,
    utilisation=0.2,
    open_credit_lines=6,
    real_estate_loans=1,
    past_due_30_59=0,
    past_due_60_89=0,
    past_due_90plus=0
))
print(r["score"], r["zone"], r["calibrated_pd"])
```

Invalid, under-18 and special-coded applications return an explicit status and no score.

## Licence and attribution
- **Code:** MIT licence (see `LICENSE`).
- **Data:** not distributed here. The Give Me Some Credit data come from a Kaggle competition and are subject to its rules. [ADD: exact data-use wording from the competition's "Data Access and Use" section, after checking it]
- **Author:** [ADD: name and contact link]
