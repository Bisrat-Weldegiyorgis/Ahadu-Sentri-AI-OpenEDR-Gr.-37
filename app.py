from fastapi import FastAPI
from features import extract_features
from model import model
from policy import decide
import actions

# Load the trained model and scaler




app = FastAPI()




# Define the input data model with all features



@app.post("/ingest")
async def ingest(event: dict):
    features = extract_features(event)
    score = model.score(features)
    decision = decide(score, features)

# Convert the incoming data to a DataFrame


 # Preprocess the data


 # Make a prediction


  # Return the prediction and probability



    if decision == "isolate":
        actions.isolate_endpoint(event)
    elif decision == "notify":
        actions.notify_admin(event, score)
    else:
        actions.allow(event)

    return {"score": score, "decision": decision}
