# Generate a basic Puppeteer script to scrape CBJ press release titles, types, sizes, and download links

puppeteer_script = """const puppeteer = require('puppeteer');
const fs = require('fs');

(async () => {
  const browser = await puppeteer.launch({ headless: 'new' });
  const page = await browser.newPage();

  // Go to the CBJ Monetary Policy Decisions page
  await page.goto('https://www.cbj.gov.jo/En/List/Monetary_Policy_Decisions', {
    waitUntil: 'networkidle2',
    timeout: 0,
  });

  // Wait for the file-info blocks to appear
  await page.waitForSelector('.file-info');

  // Scrape the content
  const data = await page.evaluate(() => {
    const results = [];
    document.querySelectorAll('.file-info').forEach(item => {
      const title = item.querySelector('h5')?.innerText.trim() || 'N/A';
      const type = item.querySelector('.type')?.innerText.trim() || 'N/A';
      const size = item.querySelector('.size')?.innerText.trim() || 'N/A';
      const link = item.querySelector('a')?.getAttribute('href') || '';
      results.push({
        title,
        type,
        size,
        link: link.startsWith('http') ? link : `https://www.cbj.gov.jo${link}`
      });
    });
    return results;
  });

  console.log("✅ Scraped:", data.length, "press releases");
  console.log(JSON.stringify(data, null, 2));

  // Optionally write to file
  fs.writeFileSync('cbj_press_releases.json', JSON.stringify(data, null, 2));

  await browser.close();
})();
"""

# Save to a .js file
file_path = "/mnt/data/cbj_scraper.js"
with open(file_path, "w") as f:
    f.write(puppeteer_script)

file_path
