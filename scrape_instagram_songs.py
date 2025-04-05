from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time
import json

def scrape_and_save():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    url = "https://socialbu.com/blog/trending-songs-on-instagram-reels"
    driver.get(url)
    time.sleep(5)

    data = []
    songs = driver.find_elements(By.TAG_NAME, "h3")

    for song in songs:
        try:
            title = song.find_element(By.TAG_NAME, "strong").text.strip()
        except:
            continue

        try:
            num_reels_element = song.find_element(By.XPATH, "./following-sibling::ul/li[strong/em[text()='# of Reels:']]")
            num_reels = num_reels_element.text.split(":")[-1].strip()
        except:
            num_reels = "N/A"

        try:
            audio_element = song.find_element(By.XPATH, "./following-sibling::ul/li[strong/em[text()='Audio:']]/a")
            audio_link = audio_element.get_attribute("href")
        except:
            audio_link = "N/A"

        data.append({
            "title": title,
            "uses": num_reels,
            "audio_link": audio_link
        })

    driver.quit()

    with open("data.json", "w", encoding="utf-8") as f:
        json.dump({"songs": data}, f, indent=2, ensure_ascii=False)

    print("✅ Data scraped and saved to data.json")

# Run the scraper
if __name__ == "__main__":
    scrape_and_save()
