FROM python:3.12-slim
LABEL maintainer = "medvedev.daff@gmail.com"

# Устанавливаем зависимости для инсталлятора Poetry
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Официальный инсталлятор Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Добавляем poetry в PATH
ENV PATH="/root/.local/bin:${PATH}"

# Создаём рабочую директорию
WORKDIR /app

# Копируем pyproject.toml и poetry.lock
COPY pyproject.toml ./
COPY poetry.lock ./

# Устанавливаем зависимости через Poetry
RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-interaction --no-ansi

# Копируем остальную часть проекта
COPY . .

# Запускаем Jupyter Notebook при старте контейнера
CMD ["poetry", "run", "jupyter", "notebook", \
    "--ip=0.0.0.0", \
    "--port=8888", \
    "--allow-root", \
    "--no-browser"]