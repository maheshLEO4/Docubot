import requests
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def scrape_webpage(url):
    try:
        response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Clean unnecessary tags
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.extract()
            
        text = soup.get_text(separator=' ', strip=True)
        return [Document(page_content=text, metadata={"source": url})]
    except Exception as e:
        return []

def scrape_urls_to_chunks(urls):
    all_docs = []
    for url in urls:
        all_docs.extend(scrape_webpage(url))
    
    if not all_docs: return None
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(all_docs)