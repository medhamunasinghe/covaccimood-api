from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
from typing import Union
from fastapi import Depends, FastAPI
from pydantic import BaseModel
import re

import sys
sys.path.insert(0, '/Users/medhamunasinghe/Desktop/ModelService/sentiment_analyzer/classifier')
from sentiment_analyzer.classifier.model import Model, get_model

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://167.71.32.233",
    "http://167.71.32.233:8000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"]
)

class SentimentRequest(BaseModel):
    text: str


class SentimentResponse(BaseModel):

    probabilities: Dict[str, float]
    sentiment: str
    confidence: float

# keywords to check the relevancy towards covid-19 vaccination
covidVaccKeywords = ['omicron','covid vaccine','covid19 vaccine','Covid19 vaccine','COVID-19 vaccine','Covid-19 vaccine','covid-19 vaccine','covid','Covid','covid19','Covid19','COVID-19','Covid-19','covid-19','covid dose','covid19 dose','covid dose','COVID-19 dose','Covid-19 dose','covid19','covid19 vaccines','COVID-19 shot','covid-19 shot','Covid-19 shot','Covid shot','COVID-19 jab','covid-19 jab','Covid-19 jab','Covid jab','second dose','second shot','second jab','first dose','first shot','first jab','covidvaccine','covid19vaccine','Covid19vaccine','COVID-19vaccine','Covid-19vaccine','covid-19vaccine','vaccine booster','Pfizer','pfizer','Moderna','moderna','Astra Zeneca','astrazeneca','Sinopharm','sinopharm','covaxin','fully vaccinated','Delta']

# keywords to check the relevancy towards any vaccination
vaccineKeywords = ['vaccine','vaccination','vaccinated']

# seperating the hashtags in to seperate words
def hashtagToWords(text):
    text = re.sub(r'#(\w+)', r'\1', text)
    return text

# check if input text word count is less than 3 
def isWordCountBelowThree(text):
    words = text.split()
    return len(words) < 3   

# values for text with less than 3 words
invalidTextValues = {
    "probabilities": {
        "positive": 0,
        "negative": 0,
        "neutral": 0
    },
    "sentiment": "Invalid! Please input text with more than two words",
    "confidence": 0
}    

@app.post("/predict", response_model=SentimentResponse)
def predict(request: SentimentRequest, model: Model = Depends(get_model)):

    # remove hashtags in the text
    text = hashtagToWords(request.text)

    if isWordCountBelowThree(text):
        response = invalidTextValues
    else:    
        # classify sentiment using the model
        sentiment, confidence, probabilities = model.predict(text)
        # check if request text contains any covid-19 vacination related word
        if any(keyword in text for keyword in covidVaccKeywords):
            # if the user text is Covid-19 vaccination relevant 
            response = {'sentiment': sentiment, 'confidence': confidence, 'probabilities': probabilities} 
        elif any(keyword in text for keyword in vaccineKeywords):
            # if the user text is any vaccination relevant 
            response = {'sentiment': sentiment + ' - Any Vaccination relevant', 'confidence': confidence, 'probabilities': probabilities} 
        else:
            # if the user text is not relevant to vaccines send sentiment with a message
            response = {'sentiment': sentiment + ' - Vaccination non-relevant', 'confidence': confidence, 'probabilities': probabilities}
    return response 

