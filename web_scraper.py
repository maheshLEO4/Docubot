import re
import time
import requests
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st

def is_playwright_available():
    """Check if Playwright is available"""
    try:
        from playwright.sync_api import sync_playwright
        return True
    except ImportError:
        return False

def install_playwright_browsers():
    """Install Playwright browsers if not already installed"""
    try:
        import subprocess
        import os
        
        # Check if browsers are already installed
        playwright_cache = os.path.expanduser("~/.cache/ms-playwright")
        if os.path.exists(playwright_cache) and os.listdir(playwright_cache):
            return True
        
        print("📦 Installing Playwright browsers (first time only)...")
        result = subprocess.run(
            ["playwright", "install", "chromium", "--with-deps"],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            print("✅ Playwright browsers installed successfully!")
            return True
        else:
            print(f"❌ Failed to install Playwright browsers: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error installing Playwright: {e}")
        return False

def extract_with_playwright(url):
    """Extract content using Playwright (works everywhere including cloud)"""
    if not is_playwright_available():
        return None, None
    
    # Try to install browsers if needed
    install_playwright_browsers()
    
    try:
        from playwright.sync_api import sync_playwright
        
        with sync_playwright() as p:
            # Launch browser in headless mode
            browser = p.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                ]
            )
            
            page = browser.new_page(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
            
            # Navigate to URL
            page.goto(url, wait_until='networkidle', timeout=30000)
            
            # Wait for React/dynamic content to load
            page.wait_for_timeout(3000)
            
            # Get title
            title = page.title()
            
            # Remove unwanted elements
            page.evaluate("""
                () => {
                    const unwanted = ['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe', 'noscript'];
                    unwanted.forEach(tag => {
                        document.querySelectorAll(tag).forEach(el => el.remove());
                    });
                }
            """)
            
            # Extract content from main containers or body
            content = page.evaluate("""
                () => {
                    const selectors = [
                        'main', 'article', '[role="main"]',
                        '.content', '.main-content', '#content',
                        '#root', '#app', '.App'
                    ];
                    
                    for (const selector of selectors) {
                        const element = document.querySelector(selector);
                        if (element && element.innerText && element.innerText.trim().length > 100) {
                            return element.innerText.trim();
                        }
                    }
                    
                    return document.body.innerText.trim();
                }
            """)
            
            browser.close()
            
            return content, title
            
    except Exception as e:
        print(f"❌ Playwright extraction failed: {e}")
        return None, None

def is_selenium_available():
    """Check if Selenium is available in the current environment"""
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        return True
    except ImportError:
        return False

def setup_selenium_driver():
    """Setup Selenium driver for JavaScript rendering"""
    if not is_selenium_available():
        return None
        
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

        try:
            from webdriver_manager.chrome import ChromeDriverManager
            from selenium.webdriver.chrome.service import Service
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
            return driver
        except:
            driver = webdriver.Chrome(options=chrome_options)
            return driver
            
    except Exception as e:
        print(f"❌ Selenium setup failed: {e}")
        return None

def extract_react_content(driver, url):
    """Specialized extraction for React/SPA websites"""
    try:
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.common.by import By
        
        driver.get(url)
        
        # Wait longer for React to render
        time.sleep(5)
        
        # Wait for any dynamic content to load
        WebDriverWait(driver, 10).until(
            lambda driver: driver.execute_script("return document.readyState") == "complete"
        )
        
        # Additional wait for React components
        time.sleep(2)
        
        title = driver.title
        
        # Special extraction for React apps
        content = driver.execute_script("""
            // Remove unwanted elements
            const unwanted = ['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe'];
            unwanted.forEach(tag => {
                const elements = document.getElementsByTagName(tag);
                Array.from(elements).forEach(el => el.remove());
            });
            
            // Look for React-specific elements or content
            const rootElement = document.getElementById('root') || 
                               document.querySelector('[data-reactroot]') ||
                               document.body;
            
            return rootElement.textContent ? rootElement.textContent.trim() : '';
        """)
        
        return content, title
        
    except Exception as e:
        print(f"❌ React extraction failed: {e}")
        # Fallback to basic extraction
        try:
            from selenium.webdriver.common.by import By
            return driver.find_element(By.TAG_NAME, "body").text, driver.title
        except:
            return None, None

def extract_with_selenium_enhanced(url):
    """Enhanced Selenium extraction with React support"""
    if not is_selenium_available():
        return None, None
        
    driver = setup_selenium_driver()
    if not driver:
        return None, None
        
    try:
        from selenium.webdriver.common.by import By
        
        # First try React-specific extraction
        content, title = extract_react_content(driver, url)
        
        if content and len(content) > 100:
            driver.quit()
            return content, title
        
        # Fallback to standard extraction
        driver.get(url)
        time.sleep(5)
        
        title = driver.title
        
        content = driver.execute_script("""
            // Remove unwanted elements
            const unwanted = ['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe', 'noscript'];
            unwanted.forEach(tag => {
                const elements = document.getElementsByTagName(tag);
                Array.from(elements).forEach(el => el.remove());
            });
            
            // Get text from common content containers
            const contentSelectors = [
                'main', 'article', '[role="main"]', 
                '.content', '.main-content', '.post-content',
                '.entry-content', '.article-content', '.page-content',
                '#content', '#main', '.article', '.post', '.body',
                '#root', '[data-reactroot]', '.App'
            ];
            
            let content = '';
            for (const selector of contentSelectors) {
                const elements = document.querySelectorAll(selector);
                for (const el of elements) {
                    if (el.textContent && el.textContent.trim().length > 50) {
                        content += ' ' + el.textContent.trim();
                    }
                }
            }
            
            // If no specific content found, use body
            if (!content.trim()) {
                content = document.body.textContent || '';
            }
            
            return content.trim();
        """)
        
        driver.quit()
        return content, title
        
    except Exception as e:
        print(f"❌ Enhanced Selenium extraction failed: {e}")
        try:
            driver.quit()
        except:
            pass
        return None, None

def extract_with_requests(url):
    """Extract content using requests + BeautifulSoup (works for server-rendered sites)"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe', 'noscript']):
            element.decompose()
        
        # Extract title
        title = soup.title.string if soup.title else url
        
        # Try to get main content from common containers
        content = None
        content_selectors = [
            ('main', {}),
            ('article', {}),
            ('div', {'role': 'main'}),
            ('div', {'class': ['content', 'main-content', 'post-content', 'entry-content', 
                              'article-content', 'page-content']}),
            ('div', {'id': ['content', 'main', 'root']}),
        ]
        
        for tag, attrs in content_selectors:
            if attrs:
                elements = soup.find_all(tag, attrs)
            else:
                elements = soup.find_all(tag)
            
            for element in elements:
                text = element.get_text(separator=' ', strip=True)
                if text and len(text) > 100:
                    content = text
                    break
            
            if content:
                break
        
        # Fallback to body
        if not content:
            content = soup.body.get_text(separator=' ', strip=True) if soup.body else ''
        
        return content, title
        
    except Exception as e:
        print(f"❌ Requests extraction failed: {e}")
        return None, None

def scrape_webpage(url):
    """
    Scrape webpage with multiple fallback methods
    Works for server-rendered sites. Client-side React apps require local Playwright.
    """
    print(f"🌐 Attempting to scrape: {url}")
    
    if 'scraping_status' not in st.session_state:
        st.session_state.scraping_status = {}
    
    # Method 1: Try requests + BeautifulSoup (works for most sites)
    st.session_state.scraping_status[url] = "Trying requests + BeautifulSoup..."
    content, title = extract_with_requests(url)
    
    if content and len(content) > 50:
        st.session_state.scraping_status[url] = "Requests + BeautifulSoup successful!"
        cleaned_content = clean_content(content)
        print(f"✅ Requests extracted {len(cleaned_content)} characters from {url}")
        return create_document(cleaned_content, url, title, "requests")
    
    # Method 2: Try Playwright (only works locally or if browsers installed)
    st.session_state.scraping_status[url] = "Trying Playwright for JavaScript content..."
    content, title = extract_with_playwright(url)
    
    if content and len(content) > 50:
        st.session_state.scraping_status[url] = "Playwright successful!"
        cleaned_content = clean_content(content)
        print(f"✅ Playwright extracted {len(cleaned_content)} characters from {url}")
        return create_document(cleaned_content, url, title, "playwright")
    
    # Method 3: Try Selenium (fallback for local environments with Chrome)
    st.session_state.scraping_status[url] = "Trying Selenium..."
    content, title = extract_with_selenium_enhanced(url)
    
    if content and len(content) > 50:
        st.session_state.scraping_status[url] = "Selenium successful!"
        cleaned_content = clean_content(content)
        print(f"✅ Selenium extracted {len(cleaned_content)} characters from {url}")
        return create_document(cleaned_content, url, title, "selenium_enhanced")
    
    # All methods failed
    st.session_state.scraping_status[url] = "Cannot scrape client-side JavaScript apps"
    print(f"❌ Failed to scrape {url}")
    print(f"💡 This appears to be a client-side JavaScript app.")
    print(f"💡 On Streamlit Cloud: Only server-rendered sites work (like news sites, blogs, documentation)")
    print(f"💡 For React/Vue/Angular apps: Run locally with Playwright/Selenium installed")
    return None

def clean_content(content):
    """Clean extracted content"""
    if not content:
        return ""
    
    # Replace multiple whitespaces with single space
    content = re.sub(r'\s+', ' ', content)
    
    # Split into sentences and filter short ones
    sentences = [sentence.strip() for sentence in content.split('.') if len(sentence.strip()) > 20]
    cleaned_content = '. '.join(sentences)
    
    return cleaned_content.strip()

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
    successful_urls = []

    if 'scraping_status' not in st.session_state:
        st.session_state.scraping_status = {}

    for url in urls:
        print(f"\n📥 Processing: {url}")
        st.session_state.scraping_status[url] = "Starting..."
        
        documents = scrape_webpage(url)

        if documents and len(documents[0].page_content) > 50:
            all_documents.extend(documents)
            successful_urls.append(url)
            print(f"✅ Successfully processed: {url}")
        else:
            print(f"❌ Failed to process: {url}")

    if not all_documents:
        print("❌ No documents were successfully processed!")
        return None

    print(f"\n📊 Successfully scraped: {len(successful_urls)}/{len(urls)} URLs")

    # Create chunks
    print("✂️ Creating text chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    text_chunks = text_splitter.split_documents(all_documents)
    print(f"📦 Created {len(text_chunks)} text chunks")

    return text_chunks