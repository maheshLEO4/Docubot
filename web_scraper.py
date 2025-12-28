import requests
from bs4 import BeautifulSoup
from typing import List, Optional, Tuple
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st

class WebScraper:
    """Optimized web scraper for cloud deployment"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
        }
    
    def scrape_url(self, url: str) -> Optional[Tuple[str, str]]:
        """Scrape single URL"""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer']):
                element.decompose()
            
            # Get title
            title = soup.title.string if soup.title else url
            
            # Get content from common containers
            content_selectors = ['main', 'article', '[role="main"]', '.content', '.main-content']
            content = None
            
            for selector in content_selectors:
                elements = soup.select(selector)
                for element in elements:
                    text = element.get_text(strip=True)
                    if len(text) > 100:
                        content = text
                        break
                if content:
                    break
            
            # Fallback to body
            if not content:
                content = soup.body.get_text(strip=True) if soup.body else ''
            
            # Clean content
            content = ' '.join(content.split())
            
            if len(content) < 50:
                return None
            
            return content, title
            
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None
    
    def scrape_urls(self, urls: List[str]) -> List[Document]:
        """Scrape multiple URLs"""
        documents = []
        successful_urls = []
        
        if not urls:
            return documents
        
        with st.spinner(f"Scraping {len(urls)} website(s)..."):
            for url in urls:
                result = self.scrape_url(url.strip())
                if result:
                    content, title = result
                    
                    doc = Document(
                        page_content=content,
                        metadata={
                            'source': url,
                            'title': title,
                            'type': 'web',
                            'scraping_method': 'requests'
                        }
                    )
                    documents.append(doc)
                    successful_urls.append(url)
        
        return documents, successful_urls
    
    def create_chunks(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        if not documents:
            return []
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        return text_splitter.split_documents(documents)