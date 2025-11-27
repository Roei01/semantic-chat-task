import requests
import json
import time
from pathlib import Path
from typing import List, Dict, Set, Optional

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


class TabuVerdictScraper:
    def __init__(self, output_dir: str = "downloads"):
        self.base_url = "https://www.gov.il"
        self.search_url = "https://www.gov.il/he/Departments/DynamicCollectors/tabu_search_verdict?skip=0"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        self.data_file = self.output_dir / "documents_data.json"


    def load_data(self) -> List[Dict]:
        if not self.data_file.exists():
            return []
        with open(self.data_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_data(self, documents: List[Dict]) -> Path:
        with open(self.data_file, "w", encoding="utf-8") as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)
        return self.data_file


    def gather_data(
        self,
        target_pdf: int = 50,
        target_word: int = 50,
        max_pages: int = 50,
        initial_seen_urls: Optional[Set[str]] = None,
        initial_pdf_count: int = 0,
        initial_word_count: int = 0,
    ) -> List[Dict]:
        chrome_options = Options()
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        driver = webdriver.Chrome(options=chrome_options)

        documents: List[Dict] = []
        pdf_count = initial_pdf_count
        word_count = initial_word_count

        base_search_url = self.search_url.partition("?")[0]
        seen_urls: Set[str] = set(initial_seen_urls or set())
        page_idx = 0

        try:
            while (pdf_count < target_pdf or word_count < target_word) and page_idx < max_pages:
                skip = page_idx * 20
                page_url = f"{base_search_url}?skip={skip}"
                print(f"Loading page {page_idx + 1}: {page_url}")

                driver.get(page_url)
                wait = WebDriverWait(driver, 10)

                try:
                    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".dy-file-item")))
                    items = driver.find_elements(By.CSS_SELECTOR, ".dy-file-item")
                except Exception:
                    print(f"No items found on page {page_idx + 1}")
                    break

                if not items:
                    break

                initial_len = len(documents)

                for item in items:
                    if pdf_count >= target_pdf and word_count >= target_word:
                        break

                    try:
                        link = item.find_element(By.TAG_NAME, "a")
                        href = link.get_attribute("href")

                        if not href or href in seen_urls:
                            continue

                        doc_type = None
                        if item.find_elements(By.CSS_SELECTOR, ".sprite-logo-pdf"):
                            doc_type = "pdf"
                        elif (
                            item.find_elements(By.CSS_SELECTOR, ".sprite-logo-docx")
                            or item.find_elements(By.CSS_SELECTOR, ".sprite-logo-doc")
                        ):
                            doc_type = "word"

                        if not doc_type:
                            continue

                        if doc_type == "pdf" and pdf_count >= target_pdf:
                            continue
                        if doc_type == "word" and word_count >= target_word:
                            continue

                        seen_urls.add(href)

                        if doc_type == "pdf":
                            pdf_count += 1
                        else:
                            word_count += 1

                        documents.append({
                            "url": href,
                            "type": doc_type,
                            "name": link.text.strip() or f"doc_{len(documents)}",
                            "downloaded": False,
                            "failed_attempts": 0,
                        })

                    except Exception:
                        continue

                print(f"Status (collected): PDF={pdf_count}/{target_pdf}, Word={word_count}/{target_word}")

                if len(documents) == initial_len and len(items) < 20:
                    break

                page_idx += 1
                time.sleep(1)

        finally:
            driver.quit()

        return documents


    def download_file(self, url: str, filename: str, max_retries: int = 3) -> bool:
        filepath = self.output_dir / filename
        last_error = None

        if filepath.exists():
            print(f"File already exists, skipping: {filename}")
            return True

        for attempt in range(1, max_retries + 1):
            try:
                print(f"  -> Download attempt {attempt} for {url}")
                response = self.session.get(url, stream=True, timeout=30)
                response.raise_for_status()

                with open(filepath, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                return True

            except Exception as e:
                last_error = e
                print(f"Failed attempt {attempt} for {url}: {e}")
                time.sleep(2 * attempt)

        print(f"Giving up on {url} after {max_retries} attempts. Last error: {last_error}")
        return False


    def download_all(self, documents: List[Dict], target_pdf: int, target_word: int):
        pdf_ok = sum(1 for d in documents if d.get("downloaded") and d["type"] == "pdf")
        word_ok = sum(1 for d in documents if d.get("downloaded") and d["type"] == "word")

        print(f"Already downloaded before run: PDF={pdf_ok}, Word={word_ok}")

        for idx, doc in enumerate(documents):
            if doc.get("downloaded"):
                continue 

            if doc["type"] == "pdf" and pdf_ok >= target_pdf:
                continue
            if doc["type"] == "word" and word_ok >= target_word:
                continue

            ext = ".pdf" if doc["type"] == "pdf" else ".docx"
            safe_name = doc["name"][:30].replace("/", "_").replace("\\", "_")
            filename = f"doc_{idx}_{safe_name}{ext}"

            print(f"Downloading {idx + 1}/{len(documents)}: {filename} ({doc['type']})")

            ok = self.download_file(doc["url"], filename)
            if ok:
                doc["downloaded"] = True
                if doc["type"] == "pdf":
                    pdf_ok += 1
                else:
                    word_ok += 1
            else:
                doc["failed_attempts"] = doc.get("failed_attempts", 0) + 1

            print(f"Status (downloaded): PDF={pdf_ok}/{target_pdf}, Word={word_ok}/{target_word}")
            time.sleep(0.5)

        self.save_data(documents)

        print("\nFinal status:")
        print(f"PDF downloaded successfully: {pdf_ok}")
        print(f"Word downloaded successfully: {word_ok}")

        failed_docs = [d for d in documents if not d.get("downloaded")]
        if failed_docs:
            print(f"Documents that still failed after retries: {len(failed_docs)}")
        else:
            print("All documents downloaded successfully (subject to availability).")


    def run(self, target_pdf: int = 50, target_word: int = 50, max_pages: int = 50):
        existing_docs = self.load_data()

        downloaded_urls = {d["url"] for d in existing_docs if d.get("downloaded")}
        pdf_ok = sum(1 for d in existing_docs if d.get("downloaded") and d["type"] == "pdf")
        word_ok = sum(1 for d in existing_docs if d.get("downloaded") and d["type"] == "word")

        remaining_pdf = max(target_pdf - pdf_ok, 0)
        remaining_word = max(target_word - word_ok, 0)

        print(f"Need to add: PDF={remaining_pdf}, Word={remaining_word}")

        if remaining_pdf > 0 or remaining_word > 0:
            new_docs = self.gather_data(
                target_pdf=remaining_pdf,
                target_word=remaining_word,
                max_pages=max_pages,
                initial_seen_urls=downloaded_urls,
                initial_pdf_count=0,
                initial_word_count=0,
            )
            existing_docs.extend(new_docs)
            self.save_data(existing_docs)

        self.download_all(existing_docs, target_pdf, target_word)


def main():
    scraper = TabuVerdictScraper(output_dir="scraper/data")
    scraper.run(target_pdf=50, target_word=50, max_pages=50)


if __name__ == "__main__":
    main()
