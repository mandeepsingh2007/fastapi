# Use an official Python runtime as a base image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy everything into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the API port
EXPOSE 7860

# Run the FastAPI app
CMD ["uvicorn", "ml_api:app", "--host", "0.0.0.0", "--port", "7860"]
