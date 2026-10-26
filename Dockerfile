# syntax=docker/dockerfile:1
# Quantum Compass: reproducible Streamlit/Qiskit runtime image.
# The image uses Python 3.12, matching runtime.txt and Streamlit Cloud guidance.
FROM python:3.12-slim-bookworm

LABEL org.opencontainers.image.title="Quantum Compass" \
      org.opencontainers.image.description="QAOA optimisation-landscape navigator" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    BROWSER_GATHER_USAGE_STATS=false \
    HOME=/home/quantum

WORKDIR /app

# libgomp is required by common NumPy/scikit-learn/Qiskit-Aer binary wheels;
# tini forwards container stop signals cleanly to Streamlit and its Aer worker.
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 tini \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin quantum

# Install dependencies before application code to keep this expensive layer cacheable.
COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

# .dockerignore prevents local credentials, caches, virtual environments, and
# generated reports from entering this image.
COPY --chown=quantum:quantum . .
RUN mkdir -p /app/data/advantage_maps /app/data/barren_diagrams \
    && chown -R quantum:quantum /app

USER quantum

# Generate a cache compatible with the exact Python/scikit-learn versions in
# this image. This runs once at image-build time, not when a user opens the app.
RUN python -c "from core.fingerprint import DEFAULT_MODEL_PATH, load_default_classifier; model = load_default_classifier(force_retrain=True, n_per_env=50); print('Prepared {} with {} classes and OOB score {:.3f}.'.format(DEFAULT_MODEL_PATH, len(model.classes_), getattr(model, '_qc_oob_score', 0.0)))"

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=5).read()" || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]
