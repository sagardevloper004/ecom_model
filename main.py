import requests


def callAnswer():
   

    url = "http://localhost:8000/search"
    payload = {
        "query": "provide a summary of the sales data for the last quarter",
        "top_k": 3
    }
    response = requests.post(url, json=payload)
    print(response.json())

if __name__ == "__main__":
    callAnswer()