import requests

OLLAMA_URL = "http://localhost:11434/api/embeddings"
test_text = "hello world"

try:
    response = requests.post(
        OLLAMA_URL,
        json={"model": "llama3", "prompt": test_text}
    )
    print("Status code:", response.status_code)
    print("Response:", response.text)
except Exception as e:
    print("Error:", e)