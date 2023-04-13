
FROM python:3.10-alpine

COPY ./requirements/requirements.txt ./requirements/requirements.txt
RUN pip install pip==23.0.1
RUN pip install -r requirements/requirements.txt
RUN pip --no-cache-dir install torch

COPY ./sentiment_analyzer /sentiment_analyzer
COPY BertCNN_bestModel.bin /

EXPOSE 8000

CMD ["uvicorn", "sentiment_analyzer.main:app", "--reload", "--host", "0.0.0.0"]