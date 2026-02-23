FROM python:3.12-slim

# Instalar uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /home/app

# Copiar definición de dependencias
COPY pyproject.toml .

# Instalar dependencias con uv (sin crear venv, directo al sistema)
RUN uv sync --no-dev

# Install Jupyter Notebook
RUN pip install jupyter

# Crear carpetas de trabajo
RUN mkdir -p models notebooks

# Exponer puerto de JupyterLab
EXPOSE 8888

# Arrancar JupyterLab
CMD ["uv", "run", "jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--notebook-dir=/home/app/notebooks", \
     "--ServerApp.token=dev123"]