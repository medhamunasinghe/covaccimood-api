
FROM python:3.9-alpine

COPY ./requirements/requirements.txt ./requirements/requirements.txt
RUN pip install pip==22.0.4
RUN pip install -r requirements/requirements.txt

COPY ./sentiment_analyzer /sentiment_analyzer
COPY BertCNN_bestModel.bin /

EXPOSE 8000

CMD ["uvicorn", "sentiment_analyzer.main:app", "--reload", "--host", "0.0.0.0"]