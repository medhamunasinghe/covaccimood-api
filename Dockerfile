
FROM python:3.7

COPY ./requirements/requirements.txt ./requirements/requirements.txt
RUN pip3 install -r requirements/requirements.txt
RUN pip3 --no-cache-dir install torch

COPY ./sentiment_analyzer /sentiment_analyzer
COPY BertCNN_bestModel.bin /

EXPOSE 8000

CMD ["uvicorn", "sentiment_analyzer.main:app", "--host", "0.0.0.0", "--port", "8000"]