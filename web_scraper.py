import re
import time
import requests
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st

def is_selenium_available():
    """Check if Selenium is available in the current environment"""
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        return True
    except ImportError:
        return False

def setup_selenium_driver():
    """Setup Selenium driver for JavaScript rendering - only if available"""
    if not is_selenium_available():
        return None
        
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

        # Try different methods to initialize driver
        try:
            from webdriver_manager.chrome import ChromeDriverManager
            from selenium.webdriver.chrome.service import Service
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
            return driver
        except:
            # Fallback to direct Chrome driver
            driver = webdriver.Chrome(options=chrome_options)
            return driver
            
    except Exception as e:
        print(f"❌ Selenium setup failed: {e}")
        return None

def extract_with_selenium(url):
    """Extract content using Selenium if available"""
    if not is_selenium_available():
        return None, None
        
    driver = setup_selenium_driver()
    if not driver:
        return None, None
        
    try:
        driver.get(url)
        # Wait for page to load
        time.sleep(3)
        
        # Get page title
        title = driver.title
        
        # Extract content using simple JavaScript
        content = driver.execute_script("""
            // Remove unwanted elements
            const unwanted = ['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe'];
            unwanted.forEach(tag => {
                const elements = document.getElementsByTagName(tag);
                for (let el of elements) {
                    el.remove();
                }
            });
            
            // Get main content from common content containers
            const contentSelectors = [
                'main', 'article', '[role="main"]', 
                '.content', '.main-content', '.post-content',
                '#content', '#main', '.article'
            ];
            
            let content = '';
            for (const selector of contentSelectors) {
                const elements = document.querySelectorAll(selector);
                for (const el of elements) {
                    if (el.textContent && el.textContent.trim().length > 100) {
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
        print(f"❌ Selenium extraction failed: {e}")
        try:
            driver.quit()
        except:
            pass
        return None, None

def extract_with_requests(url):
    """Extract content using requests and BeautifulSoup"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'meta', 'link', 'nav', 'header', 'footer', 'aside', 'iframe']):
            element.decompose()
        
        # Get title
        title = soup.title.string if soup.title else "Unknown Title"
        
        # Try to find main content areas
        content_selectors = [
            'main', 'article', '[role="main"]', 
            '.content', '.main-content', '.post-content',
            '.entry-content', '.article-content',
            '#content', '#main', '.article'
        ]
        
        main_content = None
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
            # Remove navigation and other noise
            for noise in soup.select('[class*="nav"], [class*="menu"], [class*="sidebar"], [class*="ad"]'):
                noise.decompose()
            main_content = soup.get_text()
        
        return main_content, title
        
    except Exception as e:
        print(f"❌ Requests extraction failed: {e}")
        return None, None

def clean_content(content):
    """
    Clean extracted content while preserving meaningful information
    """
    if not content:
        return ""
    
    # Remove excessive whitespace and newlines
    content = re.sub(r'\s+', ' ', content)
    
    # Remove very short lines and noise
    lines = [line.strip() for line in content.split('.') if len(line.strip()) > 30]
    cleaned_content = '. '.join(lines)
    
    # Remove common noise patterns
    noise_patterns = [
        r'cookie policy|privacy policy|terms of service',
        r'sign in|login|register|subscribe',
        r'facebook|twitter|instagram|linkedin',
        r'copyright|all rights reserved',
        r'home|about|contact|search',
    ]
    
    for pattern in noise_patterns:
        cleaned_content = re.sub(pattern, '', cleaned_content, flags=re.IGNORECASE)
    
    return cleaned_content.strip()

def scrape_webpage(url):
    """
    Scrape webpage and return LangChain documents
    """
    print(f"🌐 Attempting to scrape: {url}")
    
    # Show progress in Streamlit
    if 'scraping_status' not in st.session_state:
        st.session_state.scraping_status = {}
    
    st.session_state.scraping_status[url] = "Starting..."
    
    # Method 1: Try Selenium first (for JavaScript-heavy sites)
    st.session_state.scraping_status[url] = "Trying Selenium..."
    content, title = extract_with_selenium(url)
    
    if content and len(content) > 100:
        st.session_state.scraping_status[url] = "Selenium successful!"
        cleaned_content = clean_content(content)
        print(f"✅ Selenium extracted {len(cleaned_content)} characters from {url}")
        return create_document(cleaned_content, url, title, "selenium")
    
    # Method 2: Fallback to requests + BeautifulSoup
    st.session_state.scraping_status[url] = "Trying requests..."
    content, title = extract_with_requests(url)
    
    if content and len(content) > 100:
        st.session_state.scraping_status[url] = "Requests successful!"
        cleaned_content = clean_content(content)
        print(f"✅ Requests extracted {len(cleaned_content)} characters from {url}")
        return create_document(cleaned_content, url, title, "requests")
    
    # Method 3: Final fallback - simple requests with minimal processing
    st.session_state.scraping_status[url] = "Trying simple request..."
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        content = soup.get_text()
        cleaned_content = clean_content(content)
        title = soup.title.string if soup.title else "Unknown Title"
        
        if len(cleaned_content) > 50:
            st.session_state.scraping_status[url] = "Simple request successful!"
            print(f"⚠️ Simple request extracted {len(cleaned_content)} characters from {url}")
            return create_document(cleaned_content, url, title, "simple")
        
    except Exception as e:
        print(f"❌ All extraction methods failed for {url}: {e}")
    
    st.session_state.scraping_status[url] = "Failed to scrape"
    return [Document(
        page_content=f"Failed to load content from {url}. The website might be blocking automated access or require JavaScript.",
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
    successful_urls = []
    failed_urls = []

    # Initialize scraping status
    if 'scraping_status' not in st.session_state:
        st.session_state.scraping_status = {}

    for url in urls:
        print(f"\n📥 Processing: {url}")
        st.session_state.scraping_status[url] = "Starting..."
        
        documents = scrape_webpage(url)

        if documents and len(documents[0].page_content) > 100:
            all_documents.extend(documents)
            successful_urls.append(url)
            print(f"✅ Successfully processed: {url}")
        else:
            failed_urls.append(url)
            print(f"❌ Failed to process: {url}")

    if not all_documents:
        print("❌ No documents were successfully processed!")
        return None

    print(f"\n📊 Successfully scraped: {len(successful_urls)}/{len(urls)} URLs")
    
    if failed_urls:
        print(f"❌ Failed URLs: {failed_urls}")

    # Create chunks
    print("✂️ Creating text chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    text_chunks = text_splitter.split_documents(all_documents)
    print(f"📦 Created {len(text_chunks)} text chunks")

    return text_chunks