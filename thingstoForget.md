docker run --rm -it -v $(pwd):/app ncf-docker /bin/bash runs container
python main.py | tee /app/training_log.txt /mnt/c/Users/DaNi/Documents/DeepL/NC
docker run -it -v "$PWD":/app ncf-keras-theano bash