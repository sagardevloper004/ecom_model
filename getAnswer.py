import requests

url = "http://localhost:8000/ask"
question = "What is the total sales for product FUR-BO-10001798?"

response = requests.post(url, json={"question": question})

if response.ok:
    print("Response:", response.json())
else:
    print("Error:", response.status_code, response.text)