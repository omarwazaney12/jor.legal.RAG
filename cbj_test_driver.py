from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

options = Options()
options.add_argument("--headless")  # Optional: run in background
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

# This will automatically download & use the right ChromeDriver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

driver.get("https://www.google.com")
print("✅ Page title:", driver.title)

driver.quit()
