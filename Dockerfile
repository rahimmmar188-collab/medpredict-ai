# Use official lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Generate data and train models (for self-contained image)
# In production, models would be trained outside and just copied.
RUN cd data && python generate_mock_data.py
RUN cd src && python train.py

# Expose port
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
