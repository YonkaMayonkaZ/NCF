FROM python:3.6

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set default command to run training
CMD ["python", "main.py", "--batch_size=256", "--lr=0.001", "--factor_num=16"]
