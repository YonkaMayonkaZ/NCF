# Neural Collaborative Filtering - Experimental Setup

This project reproduces and extends the results of the Neural Collaborative Filtering (NeuMF) framework using the `u.data` dataset from MovieLens. The repository is organized to support multiple experimental conditions and reproducible execution through Docker.

## 🔧 Environment

This project uses legacy versions of deep learning frameworks:

- Python 2.7
- Keras 1.0.7
- Theano 0.8.0
- NumPy, SciPy, Matplotlib

To ensure reproducibility, a Docker image named `ncf-keras-theano` is used to encapsulate all dependencies.

### Running inside Docker

To execute any experiment:
```bash
docker run --rm -v $(pwd):/app -w /app/experiments/<experiment-folder> \
ncf-keras-theano python <script>.py > output.txt 2>&1
