import requests

url = "https://www.gov.il/he/Departments/DynamicCollectors/tabu_search_verdict?skip=0"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

try:
    response = requests.get(url, headers=headers)
    print(f"Status: {response.status_code}")
    content = response.text
    print(f"Length: {len(content)}")
    if "dynamicCtrl.ViewModel.dataResults" in content:
        print("Found Angular data reference.")
    
    if "doc_1_3-382-2024.docx" in content or "doc_0_3-382-2024.pdf" in content:
        print("Found specific filenames in source!")
    else:
        print("Did NOT find specific filenames in source.")
        
    if ".pdf" in content:
        print("Found .pdf in source")
    
except Exception as e:
    print(f"Error: {e}")
