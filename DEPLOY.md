# Deploying ProcurementGuard

Two routes. Take the first one. The second exists because the first one might
not, and Day 6 is not allowed to slip.

---

## Route A — Streamlit Community Cloud (free, ~10 minutes)

This is the one to do. Free, no card, gives you a public URL you can put on a
slide.

**1. Push to GitHub.**
```bash
git init
git add .
git commit -m "ProcurementGuard: rules + anomaly + graph, deployed"
git branch -M main
git remote add origin https://github.com/<you>/procurementguard.git
git push -u origin main
```

Make sure `.gitignore` contains `.venv/` and `__pycache__/`.

**Commit `data/`.** Normally you would not commit generated files — but here the
whole dataset is 2 MB, it makes the repo runnable by anyone in one click, and
Streamlit Cloud will not run your pipeline for you. If you would rather build it
on the server, see the note at the bottom.

**2. Deploy.**
- Go to https://share.streamlit.io
- Sign in with GitHub
- *New app* → pick the repo → main branch → main file path `app.py`
- Deploy

Two or three minutes. Then you have a URL like
`https://procurementguard.streamlit.app`.

**3. Check it.** Open the URL. Both tabs load. The Cartels tab shows the two
rings. If it does, Day 6 is done.

---

## Route B — Docker (works anywhere, incl. Render / Fly / Cloud Run)

```bash
docker build -t procurementguard .
docker run -p 8501:8501 procurementguard
# → http://localhost:8501
```

The Dockerfile runs the whole pipeline **during the build**:

```
RUN python -m src.main && python -m src.run_rules && python -m src.run_models
```

This is deliberate. If a tripwire fires — a broken invariant, a leaked feature —
those commands exit non-zero and **the image fails to build**. Bad data cannot
ship. The dashboard can only ever serve data that passed both gates.

To put it on Render:
- New → Web Service → connect the repo
- Runtime: **Docker**
- Free instance type
- Deploy

---

## If the build fails

**`ModuleNotFoundError`** → something is missing from `requirements.txt`. Test it
in a clean `.venv` first, which is what a `.venv` is *for*:
```bash
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt
python -m src.main
```
If it runs there, it runs in Docker.

**`AssertionError: LEAK` or `invariants`** → this is the tripwire doing its job.
Do not raise the ceiling. Do not delete the assert. Find what changed in the
generator.

**App loads but the tables are empty** → `data/*.csv` were not committed and the
pipeline never ran. Either commit them, or add a build step that runs the three
commands.

**Streamlit Cloud memory limit (1 GB)** → the app only reads CSVs; it trains
nothing at runtime. If you ever hit this, something is being computed on page
load that should have been computed in the pipeline.

---

## Note on committing `data/`

Purists will say generated files do not belong in git. They are right in general
and wrong here: this repo is a demo, the data is 2 MB, and "clone and it runs"
is worth more than doctrinal cleanliness. The Dockerfile builds the data from
source anyway, so both stories are true at once.
