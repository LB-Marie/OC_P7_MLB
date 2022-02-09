###############################################################################
##                              Library imports                              ##
###############################################################################
import uvicorn
from fastapi import FastAPI
import pickle
from pydantic import BaseModel
import numpy as np
import pandas as pd
#
###############################################################################
##                           Creation of the app object                      ##
###############################################################################
#
app = FastAPI()
pickle_in = open("Scoring_model.pkl","rb")
classifier=pickle.load(pickle_in)
#
###############################################################################
##         Creation of class to define the parameters for the API            ##
###############################################################################
#
class Scoring_data(BaseModel):
    EXT_SOURCE_2 : float
    EXT_SOURCE_3 : float
    EXT_SOURCE_1 : float
    DAYS_BIRTH : float
    DAYS_EMPLOYED_PERC : float
    DAYS_ID_PUBLISH : float
    DAYS_REGISTRATION : float
    BURO_DAYS_CREDIT_MEAN : float
    BURO_DAYS_CREDIT_MAX : float
    INSTAL_DAYS_ENTRY_PAYMENT_MEAN : float
    ACTIVE_DAYS_CREDIT_MAX : float
    DAYS_LAST_PHONE_CHANGE : float
    INSTAL_DAYS_ENTRY_PAYMENT_SUM : float
    BURO_DAYS_CREDIT_ENDDATE_MEAN : float
    PREV_APP_CREDIT_PERC_MEAN : float
    REGION_POPULATION_RELATIVE : float
    APPROVED_APP_CREDIT_PERC_MAX : float
    APPROVED_AMT_ANNUITY_MAX : float
    PREV_HOUR_APPR_PROCESS_START_MEAN : float
    AMT_GOODS_PRICE : float
    ACTIVE_DAYS_CREDIT_MIN : float
    CLOSED_DAYS_CREDIT_MIN : float
    ACTIVE_DAYS_CREDIT_ENDDATE_MAX : float
    POS_COUNT : float
#
###############################################################################
##    Creation of post for the API go get the probability and predictions    ##
###############################################################################
#       
@app.post('/predict')
def predict_score(data:Scoring_data):
    data = data.dict()
    values = np.array(list(data.values())).reshape(1, -1)
    prediction = int(classifier.predict(values))
    probability = round(classifier.predict_proba(values)[0,1], 2)
    return {
        'prediction': str(prediction), 
        'probability' : str(probability) }

@app.post('/predict_client')
def predict_score_from_client(level:int):
    data = pd.read_csv('P7_client_selected_data.csv')
    if level < len(data):
        d = data.iloc[level].to_frame().transpose()
        prediction = int(classifier.predict(d))
        probability = round(classifier.predict_proba(d)[0,1], 2)
        return {
            'prediction': str(prediction), 
            'probability' : str(probability) }
    else:
        return("This client is does not exist")
