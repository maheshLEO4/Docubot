import re
import time
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

def setup_selenium_driver():
    """Setup Selenium driver for JavaScript rendering"""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        return driver
    except Exception as e:
        print(f"❌ Selenium setup failed: {e}")
        print("📝 Falling back to simple requests method...")
        return None

def extract_meaningful_content(driver, url):
    """
    Extract meaningful content from any webpage
    """
    try:
        # Wait for page to load completely
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )

        # Wait for dynamic content
        time.sleep(3)

        # Extract content using JavaScript for better accuracy
        content = driver.execute_script("""
            // Remove unwanted elements
            const unwantedSelectors = [
                'script', 'style', 'meta', 'link', 'noscript',
                '[class*="comment"]', '[id*="comment"]',
                '[class*="advertisement"]', '[id*="ad"]',
                '[class*="sidebar"]', '[class*="menu"]',
                'header', 'footer', 'nav', 'aside'
            ];

            unwantedSelectors.forEach(selector => {
                const elements = document.querySelectorAll(selector);
                elements.forEach(el => el.remove());
            });

            // Get main content areas
            const contentSelectors = [
                'main', 'article', 'section',
                '[role="main"]', '.content', '.main',
                '.post', '.article', '.blog',
                '.page-content', '#content', '#main'
            ];

            let allContent = '';

            // Try to find main content first
            for (const selector of contentSelectors) {
                const elements = document.querySelectorAll(selector);
                for (const el of elements) {
                    if (el.offsetWidth > 0 && el.offsetHeight > 0) {
                        const text = el.innerText || el.textContent;
                        if (text && text.trim().length > 100) {
                            allContent += ' ' + text.trim();
                        }
                    }
                }
            }

            // If no main content found, use body but clean it
            if (!allContent.trim()) {
                const body = document.body;
                const walker = document.createTreeWalker(
                    body,
                    NodeFilter.SHOW_TEXT,
                    null,
                    false
                );

                const textNodes = [];
                let node;
                while (node = walker.nextNode()) {
                    const parent = node.parentElement;
                    if (parent &&
                        parent.offsetWidth > 0 &&
                        parent.offsetHeight > 0 &&
                        getComputedStyle(parent).visibility !== 'hidden' &&
                        getComputedStyle(parent).display !== 'none' &&
                        node.textContent.trim().length > 10) {
                        textNodes.push(node.textContent.trim());
                    }
                }
                allContent = textNodes.join(' ');
            }

            return allContent;
        """)

        return content.strip()

    except Exception as e:
        print(f"❌ Error extracting content with JavaScript: {e}")
        try:
            return driver.find_element(By.TAG_NAME, "body").text
        except:
            return ""

def clean_content(content):
    """
    Clean extracted content while preserving meaningful information
    """
    if not content:
        return ""

    # Remove excessive whitespace
    content = re.sub(r'\s+', ' ', content)

    # Remove very short lines (likely navigation or noise)
    lines = content.split('. ')
    meaningful_lines = [line.strip() for line in lines if len(line.strip()) > 20]

    cleaned_content = '. '.join(meaningful_lines)

    return cleaned_content.strip()

def get_page_title(driver):
    """Extract page title"""
    try:
        return driver.title
    except:
        return "Unknown Title"

def scrape_webpage(url):
    """
    Scrape webpage and return LangChain documents
    """
    print(f"🌐 Loading: {url}")

    try:
        # Method 1: Try Selenium first for JavaScript-heavy sites
        driver = setup_selenium_driver()
        if driver:
            driver.get(url)
            page_content = extract_meaningful_content(driver, url)
            page_title = get_page_title(driver)
            driver.quit()

            if page_content and len(page_content) > 100:
                cleaned_content = clean_content(page_content)
                print(f"✅ Selenium extracted {len(cleaned_content)} characters")
                return create_document(cleaned_content, url, page_title, "selenium")

        # Method 2: Fallback to requests + BeautifulSoup
        print("🔄 Falling back to requests method...")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove unwanted elements
        for element in soup(['script', 'style', 'meta', 'link', 'nav', 'header', 'footer', 'aside']):
            element.decompose()

        # Try to find main content
        main_content = None
        content_selectors = ['main', 'article', '[role="main"]', '.content', '.main-content', '#content']

        for selector in content_selectors:
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text(strip=True)
                if len(text) > 200:
                    main_content = text
                    break
            if main_content:
                break

        # If no main content found, use body but clean it
        if not main_content:
            main_content = soup.get_text()

        cleaned_content = clean_content(main_content)
        page_title = soup.title.string if soup.title else "Unknown Title"

        print(f"✅ Requests extracted {len(cleaned_content)} characters")
        return create_document(cleaned_content, url, page_title, "requests")

    except Exception as e:
        print(f"❌ Error loading webpage {url}: {e}")
        return [Document(
            page_content=f"Failed to load content from {url}. Error: {str(e)}",
            metadata={"source": url, "title": "Error", "method": "failed"}
        )]

def create_document(content, url, title, method):
    """Create LangChain Document object"""
    return [Document(
        page_content=content,
        metadata={
            "source": url,
            "title": title,
            "scraping_method": method,
            "content_length": len(content)
        }
    )]

def scrape_urls_to_chunks(urls):
    """
    Scrape URLs and return text chunks
    """
    if isinstance(urls, str):
        urls = [urls]

    all_documents = []

    for url in urls:
        print(f"\n📥 Processing: {url}")
        documents = scrape_webpage(url)

        if documents and len(documents[0].page_content) > 50:
            all_documents.extend(documents)
            print(f"✅ Successfully processed: {url}")
        else:
            print(f"❌ Failed to process: {url}")

    if not all_documents:
        print("❌ No documents were successfully processed!")
        return None

    print(f"\n📊 Total documents loaded: {len(all_documents)}")

    # Create chunks
    print("✂️  Creating text chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    text_chunks = text_splitter.split_documents(all_documents)
    print(f"📦 Created {len(text_chunks)} text chunks")

    return text_chunks