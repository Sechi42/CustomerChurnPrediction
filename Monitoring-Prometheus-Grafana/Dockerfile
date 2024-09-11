FROM python:3.12.4


RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip 
#copy to code directory
COPY . /code 

#set permissions

RUN chmod +x /code/src

RUN pip install --no-cache-dir --upgrade -r code/src/requirements.txt

EXPOSE 8005

WORKDIR /code/src

ENV PYTHONPATH "${PYTHONPATH}:/code/src"

CMD pip install -e .

CMD ["python","prediction_model/training_pipeline.py"]
WORKDIR /code
CMD ["python","main.py"]
