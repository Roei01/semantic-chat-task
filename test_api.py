import requests
import json

url = "https://pub-justice.openapi.gov.il/pub/moj/portal/rest/searchpredefinedapi/v1/SearchPredefinedApi/Tabu/SearchPiskeiDin"

payload = {
    "skip": 0,
    "take": 20,
    "filters": {},
}

headers = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

payloads = [
    {"skip": 0, "take": 20},
    {"Skip": 0, "Take": 20},
    {"query": "", "skip": 0, "limit": 20},
    {}
]

for p in payloads:
    try:
        print(f"Testing payload: {p}")
        response = requests.post(url, json=p, headers=headers, timeout=10)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("Success!")
            print(json.dumps(data, indent=2)[:500])
            break
        else:
            print(f"Response: {response.text[:200]}")
    except Exception as e:
        print(f"Error: {e}")
