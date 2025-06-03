FROM pytorch/pytorch:1.0.1-cuda10.0-cudnn7-devel

# Set the working directory
WORKDIR /app

# Copy all project files first
COPY . .

# Install dependencies and the package in editable mode
RUN pip install --upgrade pip \
 && pip install -r requirements.txt \
 && pip install -e .

# Make src/ available in all scripts (without PYTHONPATH hacks)
ENV PYTHONPATH=/app

# Default shell
CMD ["/bin/bash"]