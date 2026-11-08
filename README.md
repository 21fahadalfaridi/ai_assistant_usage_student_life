# AI Assistant Usage in Student Life: Analysis & Reuse Prediction

Analysis of 10,000 student sessions with an AI assistant (student level, discipline, task type, session length, prompts, AI assistance level, final outcome, satisfaction) and a test of what predicts whether the student **uses the assistant again**.

## Key findings
- **Reuse is 70.6%**, so a model that always predicts "reuse" is already 70.6% accurate. Plain accuracy (and the F1 of the positive class, ~0.83 even for that dummy) is misleading here, so models are judged by ROC AUC and balanced accuracy against a dummy baseline.
- The **only variable clearly linked to reuse is `FinalOutcome`**: about 80% reuse after "Assignment Completed" / "Idea Drafted" vs about 40% after "Confused" / "Gave Up" (Cramér's V = 0.37).
- Student level, discipline, task type, session length, prompts, AI assistance level and **satisfaction show almost no relation to reuse**.
- Satisfaction follows the AI assistance level (Spearman 0.77) but does not translate into reuse.
- Best models reach **CV ROC AUC = 0.68** (accuracy 0.75 vs 0.71 baseline), and virtually all of that comes from `FinalOutcome`. Using only pre-session features the AUC is 0.53 (chance level).

## What this project does
- Data quality checks and a justified **no outlier removal** decision: the IQR rule would delete 861 valid rows (8.6%), including *every* session with AI assistance level 1 on a 1-5 rating scale.
- EDA of usage patterns, satisfaction drivers, and reuse rate per group with 95% confidence intervals.
- Statistical tests: chi-square with Cramér's V (categorical) and Spearman (numeric).
- Leak-aware modelling: `Pipeline`, stratified 5-fold CV, **dummy baseline**, three feature sets (all / pre-session only / `FinalOutcome` only), then a held-out test and permutation importance.

## Limitations
The absence of relationships between most variables suggests the data may be simulated rather than real behaviour. Results are correlational, and `FinalOutcome` is only known after a session, so it cannot predict reuse in advance.

## Run it
```bash
pip install -r requirements.txt
jupyter notebook ai_assistant_usage_analysis.ipynb
```
Data: `data/ai_assistant_usage_student_life.csv`
