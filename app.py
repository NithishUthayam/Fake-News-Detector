import streamlit as st
import re
import spacy
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spacy model (small English model for efficiency)
nlp = spacy.load("en_core_web_sm", disable=["parser"])  # Disable parser for speed

def preprocess_text(text):
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Process text with spacy
    doc = nlp(text)
    
    # Keep tokens that are alphanumeric or part of named entities, lemmatize, and remove stopwords
    tokens = [
        token.lemma_.lower() 
        for token in doc 
        if (token.is_alpha or token.ent_type_)  # Keep alphabetic tokens or named entities
        and not token.is_stop  # Remove stopwords
        and not token.is_punct  # Remove punctuation
        and len(token.lemma_) > 1  # Remove single characters
    ]
    
    # Join tokens back into a string
    return " ".join(tokens)

def generate_news_query(text):
    # Process text with spacy to extract key terms
    doc = nlp(text)
    
    # Extract named entities and nouns
    terms = [token.text.lower() for token in doc if token.ent_type_ or token.pos_ == "NOUN"]
    
    # Add context for political roles if person entity detected
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            terms.append("prime minister")  # Contextual keyword
    
    # If no terms, fall back to first few words
    if not terms:
        terms = [token.text.lower() for token in doc[:5] if not token.is_stop and not token.is_punct]
    
    # Join terms, limit to 5 terms for concise query
    query = " ".join(list(dict.fromkeys(terms))[:5])  # Remove duplicates
    return query if query else "general news"  # Fallback

def get_bing_news_content(query):
    try:
        # Encode query for URL
        query = quote(query)
        url = f"https://www.bing.com/news/search?q={query}"
        
        # Fetch Bing News search page without custom User-Agent
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for bad status codes
        
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract news snippets (typically in 't_s' class)
        snippets = soup.find_all('div', class_='t_s')
        if not snippets:
            return None
        
        # Aggregate snippet texts, limit to ~5000 chars
        content = ""
        for snippet in snippets[:10]:  # Limit to top 10 results
            text = snippet.get_text(strip=True)
            if text and len(content) + len(text) <= 5000:
                content += text + " "
        
        return content.strip() if content else None
    except Exception as e:
        st.error(f"Error fetching Bing News: {str(e)}")
        return None

def compute_similarity(text1, text2):
    # Preprocess both texts
    text1 = preprocess_text(text1)
    text2 = preprocess_text(text2)
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    
    # Compute cosine similarity
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return similarity

def check_entity_context(text, news_content):
    # Check if input text mentions a person in a leadership role
    doc = nlp(text.lower())
    for ent in doc.ents:
        if ent.label_ == "PERSON" and ("prime minister" in text.lower() or "pm" in text.lower()):
            # Check if news content mentions the person in a leadership context
            if ent.text.lower() in news_content.lower() and ("prime minister" in news_content.lower() or "pm" in news_content.lower()):
                return True
    return False

def main():
    st.title("Fake News Detection with Bing News Verification")
    st.write("Enter text to verify if it is True or False News.")

    # Text input
    txt = st.text_area("Input Text", height=150, key="txt_input")

    if st.button("Predict"):
        if not txt:
            st.warning("Please enter some text to predict.")
            return

        # Display entered text
        st.subheader("Entered Text:")
        st.write(txt)

        # Generate news query from input text
        news_query = generate_news_query(txt)

        # Fetch and process Bing News content
        st.subheader("Bing News Verification:")
        news_content = get_bing_news_content(news_query)
        if news_content:
            
            
            # Compute similarity
            try:
                similarity = compute_similarity(txt, news_content)
                st.write(f"Similarity Score: {similarity:.2f}")
                
                # Check entity context for borderline cases
                entity_match = check_entity_context(txt, news_content)
                
                # Determine truthfulness with lower threshold and entity check
                threshold = 0.3  # Lowered for short inputs
                st.subheader("Truthfulness Assessment:")
                if similarity >= threshold or entity_match:
                    confidence = min(similarity * 100, 100)
                    st.markdown(f'<p style="color:#00FF00;">The given NEWS is TRUE.</p>', unsafe_allow_html=True)
                else:
                    confidence = min((1 - similarity) * 100, 100)
                    st.markdown(f'<p style="color:#FF0000;">The given NEWS is FALSE.</p>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error computing similarity: {str(e)}")
        else:
            st.warning(f"No news content found for automatically generated query.")
            st.subheader("Truthfulness Assessment:")
            st.markdown('<p style="color:#FF0000;">Unable to verify: No news content found. Try a different input text.</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()