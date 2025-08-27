from fastapi import FastAPI
from features import extract_features
from model import model
from policy import decide
import actions

app = FastAPI()

@app.post("/ingest")
async def ingest(event: dict):
    features = extract_features(event)
    score = model.score(features)
    decision = decide(score, features)

    if decision == "isolate":
        actions.isolate_endpoint(event)
    elif decision == "notify":
        actions.notify_admin(event, score)
    else:
        actions.allow(event)

    return {"score": score, "decision": decision}
