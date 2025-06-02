FROM pytorch/pytorch:1.0.1-cuda10.0-cudnn7-devel
WORKDIR /app
COPY setup.py .
COPY requirements.txt .
COPY src/ ./src/
RUN pip install -r requirements.txt
COPY . .
CMD ["/bin/bash"]