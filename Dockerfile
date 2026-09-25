FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/appuser \
    XDG_CACHE_HOME=/tmp/.cache \
    NUMBA_CACHE_DIR=/tmp/numba \
    MPLCONFIGDIR=/tmp/matplotlib \
    MPLBACKEND=Agg \
    GRADIO_TEMP_DIR=/tmp/gradio \
    APP_DATA_DIR=/home/appuser/app/project-vol \
    PORT=7860

RUN useradd --create-home --uid 1000 --shell /bin/bash appuser

WORKDIR /home/appuser/app

COPY --chown=1000:1000 requirements.txt ./
RUN python -m pip install -U pip && python -m pip install -r requirements.txt

COPY --chown=1000:1000 xenium_leiden ./xenium_leiden
COPY --chown=1000:1000 main.py ./main.py

RUN mkdir -p "$APP_DATA_DIR" /tmp/.cache /tmp/numba /tmp/matplotlib /tmp/gradio \
    && chown -R 1000:1000 /home/appuser /tmp/.cache /tmp/numba /tmp/matplotlib /tmp/gradio

USER 1000:1000

EXPOSE 7860

CMD ["python", "main.py"]
