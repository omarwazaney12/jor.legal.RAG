import requests
from bs4 import BeautifulSoup

url = "https://www.cbj.gov.jo/En/List/Monetary_Policy_Decisions"
headers = {
    "User-Agent": "Mozilla/5.0"
}

response = requests.get(url, headers=headers)
print("✅ Page loaded, status:", response.status_code)
print("➡️ Parsing HTML...")

if response.status_code == 200:
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Each item is inside a div with class 'file-info'
    entries = soup.select(".file-info")
    print(f"🔍 Found {len(entries)} press releases.\n")

    for entry in entries:
        title = entry.select_one("h5").get_text(strip=True) if entry.select_one("h5") else "N/A"
        file_type = entry.select_one(".type").get_text(strip=True) if entry.select_one(".type") else "N/A"
        size = entry.select_one(".size").get_text(strip=True) if entry.select_one(".size") else "N/A"
        link_tag = entry.find("a")
        download_link = f"https://www.cbj.gov.jo{link_tag['href']}" if link_tag and link_tag.get("href") else "N/A"

        print({
            "title": title,
            "type": file_type,
            "size": size,
            "link": download_link
        })

else:
    print("❌ Failed to fetch the page.")
