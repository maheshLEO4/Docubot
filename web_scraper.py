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
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

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
        time.sleep(5)
        
        # Get page title
        title = driver.title
        
        # Extract content using simple JavaScript
        content = driver.execute_script("""
            // Remove unwanted elements
            const unwanted = ['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe', 'noscript'];
            unwanted.forEach(tag => {
                const elements = document.getElementsByTagName(tag);
                for (let el of elements) {
                    el.remove();
                }
            });
            
            // Remove elements with common noise classes
            const noiseClasses = ['nav', 'menu', 'sidebar', 'ad', 'banner', 'cookie', 'popup', 'modal'];
            noiseClasses.forEach(className => {
                const elements = document.querySelectorAll(`[class*="${className}"]`);
                elements.forEach(el => el.remove());
            });
            
            // Get main content from common content containers
            const contentSelectors = [
                'main', 'article', '[role="main"]', 
                '.content', '.main-content', '.post-content',
                '.entry-content', '.article-content', '.page-content',
                '#content', '#main', '.article', '.post', '.body'
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
            
            // If no specific content found, use body but clean it more
            if (!content.trim()) {
                // Try to get text from visible elements only
                const allElements = document.body.querySelectorAll('*');
                const visibleTexts = [];
                
                for (const el of allElements) {
                    const style = window.getComputedStyle(el);
                    if (style.display !== 'none' && 
                        style.visibility !== 'hidden' &&
                        el.offsetParent !== null &&
                        el.textContent && 
                        el.textContent.trim().length > 10) {
                        visibleTexts.push(el.textContent.trim());
                    }
                }
                content = visibleTexts.join(' ');
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
    """Extract content using requests and BeautifulSoup with better headers"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0',
        }
        
        # Add timeout and handle redirects
        session = requests.Session()
        session.headers.update(headers)
        
        response = session.get(url, timeout=20, allow_redirects=True)
        response.raise_for_status()
        
        # Check if content is HTML
        content_type = response.headers.get('content-type', '').lower()
        if 'text/html' not in content_type:
            return None, f"Non-HTML content: {content_type}"
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove unwanted elements more aggressively
        unwanted_tags = ['script', 'style', 'meta', 'link', 'nav', 'header', 'footer', 'aside', 'iframe', 'noscript']
        for element in soup(unwanted_tags):
            element.decompose()
        
        # Remove elements with noise classes
        noise_selectors = [
            '[class*="nav"]', '[class*="menu"]', '[class*="sidebar"]', 
            '[class*="ad"]', '[class*="banner"]', '[class*="cookie"]',
            '[class*="popup"]', '[class*="modal"]', '[class*="alert"]',
            '[id*="nav"]', '[id*="menu"]', '[id*="sidebar"]'
        ]
        
        for selector in noise_selectors:
            for element in soup.select(selector):
                element.decompose()
        
        # Get title
        title = soup.title.string if soup.title else "Unknown Title"
        
        # Try to find main content areas with more selectors
        content_selectors = [
            'main', 'article', '[role="main"]', 
            '.content', '.main-content', '.post-content',
            '.entry-content', '.article-content', '.page-content',
            '.story-content', '.text-content', '.body-content',
            '#content', '#main', '.article', '.post', '.body',
            'section', '.section', '[class*="content"]'
        ]
        
        main_content = None
        for selector in content_selectors:
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text(strip=True)
                if len(text) > 100:  # Lower threshold to catch more content
                    main_content = text
                    break
            if main_content:
                break
        
        # If no main content found, use body but clean it more aggressively
        if not main_content:
            # Remove more noise from body
            for noise in soup.select('[class*="btn"], [class*="button"], [class*="link"], [class*="social"]'):
                noise.decompose()
            
            # Get text and filter out very short lines
            all_text = soup.get_text()
            lines = [line.strip() for line in all_text.split('\n') if len(line.strip()) > 20]
            main_content = '\n'.join(lines)
        
        return main_content, title
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed for {url}: {e}")
        return None, f"Request failed: {str(e)}"
    except Exception as e:
        print(f"❌ Error parsing {url}: {e}")
        return None, f"Parsing error: {str(e)}"

def clean_content(content):
    """
    Clean extracted content while preserving meaningful information
    """
    if not content:
        return ""
    
    # Remove excessive whitespace and newlines
    content = re.sub(r'\s+', ' ', content)
    
    # Remove very short lines and noise
    sentences = [sentence.strip() for sentence in content.split('.') if len(sentence.strip()) > 25]
    cleaned_content = '. '.join(sentences)
    
    # Remove common noise patterns
    noise_patterns = [
        r'cookie policy|privacy policy|terms of service|terms and conditions',
        r'sign in|login|register|subscribe|newsletter',
        r'facebook|twitter|instagram|linkedin|youtube|pinterest',
        r'copyright|all rights reserved|©',
        r'home|about|contact|search|menu|navigation',
        r'click here|learn more|read more|download now',
    ]
    
    for pattern in noise_patterns:
        cleaned_content = re.sub(pattern, '', cleaned_content, flags=re.IGNORECASE)
    
    # Remove extra spaces
    cleaned_content = re.sub(r' +', ' ', cleaned_content)
    
    return cleaned_content.strip()

def scrape_webpage(url):
    """
    Scrape webpage and return LangChain documents
    """
    print(f"🌐 Attempting to scrape: {url}")
    
    # Update scraping status
    if 'scraping_status' not in st.session_state:
        st.session_state.scraping_status = {}
    
    # Method 1: Try requests first (more reliable for most sites)
    st.session_state.scraping_status[url] = "Trying requests method..."
    content, title = extract_with_requests(url)
    
    if content and len(content) > 50:
        st.session_state.scraping_status[url] = "Requests successful!"
        cleaned_content = clean_content(content)
        print(f"✅ Requests extracted {len(cleaned_content)} characters from {url}")
        return create_document(cleaned_content, url, title, "requests")
    
    # Method 2: Try Selenium for JavaScript-heavy sites
    st.session_state.scraping_status[url] = "Trying Selenium (JavaScript)..."
    content, title = extract_with_selenium(url)
    
    if content and len(content) > 50:
        st.session_state.scraping_status[url] = "Selenium successful!"
        cleaned_content = clean_content(content)
        print(f"✅ Selenium extracted {len(cleaned_content)} characters from {url}")
        return create_document(cleaned_content, url, title, "selenium")
    
    # Method 3: Final fallback - very simple request
    st.session_state.scraping_status[url] = "Trying simple request..."
    try:
        response = requests.get(url, timeout=15, headers={
            'User-Agent': 'Mozilla/5.0 (compatible; Bot)'
        })
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            # Remove only script and style
            for script in soup(["script", "style"]):
                script.decompose()
            
            content = soup.get_text()
            cleaned_content = clean_content(content)
            title = soup.title.string if soup.title else "Unknown Title"
            
            if len(cleaned_content) > 30:
                st.session_state.scraping_status[url] = "Simple request successful!"
                print(f"⚠️ Simple request extracted {len(cleaned_content)} characters from {url}")
                return create_document(cleaned_content, url, title, "simple")
        
    except Exception as e:
        print(f"❌ Simple request also failed for {url}: {e}")
    
    # All methods failed
    error_msg = f"All extraction methods failed. Site may block bots or require authentication."
    st.session_state.scraping_status[url] = "Failed - site may block bots"
    print(f"❌ All methods failed for {url}")
    return [Document(
        page_content=error_msg,
        metadata={"source": url, "title": "Failed to Load", "method": "failed", "error": error_msg}
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

        if documents and len(documents[0].page_content) > 50 and not documents[0].page_content.startswith("All extraction methods failed"):
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