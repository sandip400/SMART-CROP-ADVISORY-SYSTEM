FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV and other libs
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Set environment variables for Django
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Expose port
EXPOSE 8000

# Run migrations and start server
# Note: Using manage.py runserver for simplicity, 
# for production use gunicorn: CMD ["gunicorn", "--bind", "0.0.0.0:8000", "ssga.wsgi:application"]
CMD python ssga/manage.py migrate && python ssga/manage.py runserver 0.0.0.0:8000
