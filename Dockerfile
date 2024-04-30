# Use Python 3.11
FROM python:3.11

# Set the working directory
WORKDIR /app

# Copy the poetry.lock and pyproject.toml files
COPY poetry.lock pyproject.toml /app/

# Install Poetry
RUN pip install poetry

# Install dependencies using Poetry
RUN poetry config virtualenvs.create false && poetry install --no-root

# Copy the rest of the application code
COPY . /app

# Expose port 3001
EXPOSE 3001

# Command to run the application
CMD ["poetry", "run", "python", "src/serve/api.py"]