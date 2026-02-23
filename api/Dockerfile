FROM python:3.12-slim

# Instalar uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /home/app

# Copiar definición de dependencias
COPY pyproject.toml .

# Instalar dependencias
RUN uv sync --no-dev

# Copiar código de la API
COPY main.py .

# Crear carpeta de modelos (será sobreescrita por el volumen)
RUN mkdir -p models

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]