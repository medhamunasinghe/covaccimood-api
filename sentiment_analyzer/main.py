from typing import Dict
from typing import Union
from fastapi import Depends, FastAPI
from pydantic import BaseModel

import sys
sys.path.insert(0, './sentiment_analyzer/classifier')
from sentiment_analyzer.classifier.model import Model, get_model

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1:8000"
]

# app.add_middleware(
#     # CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["POST"],
#     allow_headers=["*"],
# )

class SentimentRequest(BaseModel):
    text: str


class SentimentResponse(BaseModel):

    probabilities: Dict[str, float]
    sentiment: str
    confidence: float

# keywords to check the relevancy towards covid-19 vaccination
keywords = ['omicron', 'vaccine','dose','got','covid19','vaccines','vaccinated','shot','second','first','covidvaccine','COVID-19','covid-19','Covid-19','Covid','infection','effect','vaccine booster','spread','proof of vaccination','disease control','Pfizer','pfizer','Moderna','moderna','Astra Zeneca','astrazeneca','Sinopharm','sinopharm','covaxin','fully vaccinated','jab','confirmed case','Delta','prevention','unvaccinated','immunity','mask','treatment','antibody','confirmed case','severe symptom','breakthrough infection','death', 'inoculation']

# dictionary for probability values of irrelevant text request
irrelevant_dict: Dict[str, float] = {'negative': 0, 'neutral': 0, 'positive': 0}

@app.post("/predict", response_model=SentimentResponse)
def predict(request: SentimentRequest, model: Model = Depends(get_model)):
    # classify sentiment using the model
    sentiment, confidence, probabilities = model.predict(request.text)
    # check if request text contains any covid-19 vacination related word
    if any(keyword in request.text for keyword in keywords):
        # if the user text is relevant 
        response = {'sentiment': sentiment, 'confidence': confidence, 'probabilities': probabilities} 
    else:
        # if the user text is not relevant send sentiment with a message
        response = {'sentiment': sentiment + ' but not relevant for Covid-19 vaccination', 'confidence': confidence, 'probabilities': probabilities}
    return response 
