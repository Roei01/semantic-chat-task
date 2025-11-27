from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time

def get_page_source():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    
    driver = webdriver.Chrome(options=options)
    try:
        url = "https://www.gov.il/he/Departments/DynamicCollectors/tabu_search_verdict?skip=0"
        print(f"Fetching {url}")
        driver.get(url)
        time.sleep(5) # Wait for JS to load
        
        with open("page_source.html", "w", encoding="utf-8") as f:
            f.write(driver.page_source)
        print("Saved page_source.html")
        
    finally:
        driver.quit()

if __name__ == "__main__":
    get_page_source()

