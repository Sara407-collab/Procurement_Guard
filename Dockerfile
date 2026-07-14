# ProcurementGuard — the whole thing, in a box.
#
# Build order matters. requirements.txt is copied and installed BEFORE the source
# is copied, so that editing a .py file does not invalidate the layer that took
# four minutes to install xgboost.
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY app.py .
COPY .streamlit/ ./.streamlit/

# Build the data INSIDE the image. The dashboard serves results; it does not
# compute them. If any tripwire fires — a leaked feature, a broken invariant —
# these commands exit non-zero and THE BUILD FAILS. Bad data cannot ship.
RUN python -m src.main \
 && python -m src.run_rules \
 && python -m src.run_models

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
