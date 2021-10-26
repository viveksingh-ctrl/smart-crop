from fastapi.testclient import TestClient
from trial_eval import app
import json
# initalizing the app
client = TestClient(app)

# Input payload
payload = {"image_path": "test/img3.jpeg"}
# input_dir = 'test/img3.jpeg'
# Sending post requeset and capturing json response
response = client.post("/crop_image/", json=payload)
output = response.json()
print(output)

def test_response_code():
    assert response.status_code == 200

def test_validity():
    assert len(output)>2

