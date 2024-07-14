import requests
import os

API_URL = "https://autoderm.firstderm.com/v1/query"
API_KEY = os.getenv("oEpwcN1fJAROayGDESL5hVAOCEmGGMuvjrzU-rMSw9k")

# open the test image and read the bytes
with open(r"uploadedimages\acne.jpg", "rb") as f:
    image_contents = f.read()

# send the query
response = requests.post(
    API_URL,
    headers={"Api-Key": API_KEY},
    files={"file": image_contents},
    params={"language": "en", "model": "autoderm_v3_0"},
)

# get the JSON data returned
data = response.json()
print(data)

# Check if the 'predictions' key exists in the response
if 'predictions' in data:
    # Get the predictions
    predictions = data["predictions"]
    print(predictions)
else:
    print("No 'predictions' key in the response.")
    # Debug: Print the entire response
    print(response.text)
