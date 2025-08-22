# Fake News Detection with Bing News Verification

This project is a Streamlit-based application that helps verify the authenticity of news content using Bing News. It leverages NLP techniques, TF-IDF similarity, and entity checks to determine whether a given piece of news is likely to be true or false.

## Features
- **Text Preprocessing:** Cleans and lemmatizes input text using spaCy.
- **Keyword Extraction:** Automatically generates a search query based on named entities and nouns.
- **Bing News Scraping:** Fetches related news snippets from Bing News.
- **Similarity Scoring:** Computes cosine similarity between input text and news snippets using TF-IDF.
- **Entity Context Check:** Special handling for leadership roles (e.g., Prime Minister).
- **Streamlit Interface:** Simple and interactive UI to input text and view results.

## Tech Stack
- **Python**
- **Streamlit** (Frontend/UI)
- **spaCy** (NLP processing)
- **scikit-learn** (TF-IDF and cosine similarity)
- **BeautifulSoup & Requests** (Web scraping)

## Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/fake-news-detector.git
   cd fake-news-detector
   ```
2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   If `requirements.txt` is not created yet, install the required packages:
   ```bash
   pip install streamlit spacy requests beautifulsoup4 scikit-learn
   python -m spacy download en_core_web_sm
   ```

## Usage
1. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```
2. **Input news text** in the text area.
3. **Click "Predict"** to check if the news is true or false.
4. View similarity score, context checks, and results.

## How It Works
1. **Input Processing:**
   - Removes URLs and punctuation.
   - Lemmatizes and filters tokens using spaCy.
2. **Query Generation:**
   - Extracts key terms (nouns, named entities).
   - Builds a Bing News search query.
3. **News Retrieval:**
   - Scrapes top news snippets using BeautifulSoup.
4. **Verification:**
   - Calculates TF-IDF cosine similarity.
   - Performs entity context checks.
   - Displays assessment based on threshold.

## Limitations
- Dependent on Bing News search results (may vary over time).
- Not a replacement for professional fact-checking.
- Web scraping may break if Bing changes its page structure.

## Future Enhancements
- Add multiple search sources (e.g., Google News, APIs).
- Improve entity and context detection with larger NLP models.
- Deploy as a web service or integrate with messaging platforms.

## License
This project is licensed under the MIT License.

## Disclaimer
This tool is intended for educational and experimental purposes only. Always cross-check information from credible sources.

