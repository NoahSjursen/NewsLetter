import requests
import json
import datetime
import os
import shutil
from bs4 import BeautifulSoup
import google.generativeai as genai
from dotenv import load_dotenv
import re
import random
import time

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")  

# Constants
TEMP_DIR = "temp"
EMAILS_DIR = "emails"
SEARCH_ENGINE = "google_scholar"
SEARCH_LANGUAGE = "en"
NUM_RESULTS = 8
SUMMARY_WORD_LIMIT = 400
EMAIL_TEMPLATE_PATH = "emailtemplate.txt"  # Make sure to provide the correct path

# Gemini API Configuration
genai.configure(api_key=GEMINI_API_KEY)
generation_config = {
    "temperature": 0.01,
    "top_p": 0.95,
    "top_k": 32,
    "max_output_tokens": 1024,
}
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]
model = genai.GenerativeModel(
    model_name="gemini-1.5-pro-latest",
    generation_config=generation_config,
    safety_settings=safety_settings,
)

def find_papers(api_key, query, date_from=None):
    """Fetches Google Scholar results with optional date filter."""
    params = {
        "engine": SEARCH_ENGINE,
        "q": query,
        "hl": SEARCH_LANGUAGE,
        "api_key": api_key,
        "num": NUM_RESULTS,
    }
    if date_from:
        params["scisbd"] = 2
        params["as_ylo"] = date_from.year
    response = requests.get("https://serpapi.com/search", params=params)
    response.raise_for_status()
    return response.json()

def download_content(url, file_format):
    """Downloads the content of a URL and saves it to a file in the 'temp' folder."""
    try:
        # Define a list of common User-Agents 
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.5845.110 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.5845.110 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.5845.110 Safari/537.36"
        ]
        # Choose a random User-Agent from the list
        random_user_agent = random.choice(user_agents)
        headers = {'User-Agent': random_user_agent}

        response = requests.get(url, headers=headers)
        response.raise_for_status()

        os.makedirs(TEMP_DIR, exist_ok=True)

        if file_format == 'html':
            with open(os.path.join(TEMP_DIR, f"{url.split('/')[-1]}.html"), "w", encoding="utf-8") as file:
                soup = BeautifulSoup(response.text, 'html.parser')
                text_content = soup.find_all(['p', 'article', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
                text = '\n'.join([t.get_text() for t in text_content])
                file.write(text)
        elif file_format == 'pdf':
            with open(os.path.join(TEMP_DIR, f"{url.split('/')[-1]}.pdf"), "wb") as file:
                file.write(response.content)
        else:
            print(f"Unsupported file format: {file_format}")

        print(f"Downloaded {file_format} file from: {url} to temp folder")
    except Exception as e:
        print(f"An error occurred while downloading {url}: {e}")

def generate_summary(text, title, link, snippet, publication_info):
    """Generates a summary using the Gemini model and formats it."""
    prompt = f"""
    Create a {SUMMARY_WORD_LIMIT}-word summary of the following article, focusing on the key findings and implications. 

    **Formatting Rules:**

    * Use bullet points to list key findings.
    * Use bold text to highlight important terms.
    * Keep sentences concise and clear.
    * Maintain a neutral and objective tone.
    * Do not include any personal opinions or interpretations.

    **Article Text:**

    {text}
    """
    summary = model.generate_content(prompt).text

    # Add metadata to the summary
    summary += f"\n\nTitle: {title}"
    summary += f"\nLink: {link}"
    summary += f"\nSnippet: {snippet}"
    summary += f"\nPublication Info: {publication_info}"
    return summary

def create_email_content(summary, title, link, snippet, publication_info):
    print("creating email")
    """Uses the email template and populates it with the summary."""
    try:
        with open(EMAIL_TEMPLATE_PATH, "r") as f:
            email_template = f.read()

        # Combined prompt for summary and styling
        email_prompt = f"""
            Populate the following email template with the provided information and style it according to the instructions below, ensuring all HTML tags are properly closed.

            **Title:** {title}
            **Link:** {link}
            **Snippet:** {snippet} 
            **Publication Info:** {publication_info}
            **Summary:** {summary}

            **Email Template:** 

            {email_template}

            **Instructions:**
            * Replace the text between the double curly braces (`{{}}`) in the email template with the corresponding information provided above. 
            * For example, replace `{title}` with the value of the **Title** variable.
            * If any tags are missing closing tags, add them to ensure the HTML is well-formed.
            * The email should have a modern and professional look. 
            * Use a light grey background color and a white container. 
            * The title should be in a blue header with white text.
            * The content should have a dark grey heading and grey paragraph text.
            * Include a light grey footer with a blue link color.

            Return ONLY the styled HTML code.
        """
        response = model.generate_content(email_prompt)
        styled_email_html = response.text

        print("Generated Styled Email HTML:")  # For debugging
        print(styled_email_html) 

        return styled_email_html

    except Exception as e:
        print(f"An error occurred while creating the email: {e}")
        return None  # Or handle the error differently

def process_search_results(results):
    """Processes each search result: downloads, summarizes, and creates email content."""
    os.makedirs(TEMP_DIR, exist_ok=True) 
    os.makedirs(EMAILS_DIR, exist_ok=True) 

    for result in results["organic_results"]:
        title = result.get("title", "")
        link = result.get("link", "")
        snippet = result.get("snippet", "")
        publication_info = result.get("publication_info", {}).get("summary", "")

        print(f"Processing: {title}")

        # Download files
        if "link" in result:
            download_content(result["link"], "html")  
        if "resources" in result:
            for resource in result.get("resources", []):
                if resource.get("file_format") == "PDF":
                    download_content(resource["link"], "pdf")

        # Summarize downloaded content
        for filename in os.listdir(TEMP_DIR):
            if filename.endswith(".html"):
                filepath = os.path.join(TEMP_DIR, filename)
                if os.path.exists(filepath):
                    with open(filepath, "r", encoding="utf-8") as file:
                        text = file.read()
                        summary = generate_summary(text, title, link, snippet, publication_info)
                        email_content = create_email_content(summary, title, link, snippet, publication_info)

                        # Create a unique filename using the title
                        output_filename = f"{title.replace(' ', '_')}_email.html" 
                        output_filepath = os.path.join(EMAILS_DIR, output_filename)
                        with open(output_filepath, "w", encoding="utf-8") as output_file:
                            output_file.write(email_content)
            else:
                print(f"Skipping file: {filename} (not an HTML file)")

def expand_query(user_input):
    """Expands a user's input to create a more specific search query."""
    expanded_terms = {
        "neuroscience": ["neurobiology", "cognitive neuroscience", "computational neuroscience", "neuroimaging", "neurogenetics"],
        "machine learning": ["deep learning", "reinforcement learning", "natural language processing", "computer vision", "machine learning algorithms"],
        "data science": ["data analysis", "data mining", "big data", "data visualization", "predictive modeling"],
        "artificial intelligence": ["machine learning", "deep learning", "natural language processing", "computer vision", "robotics"],
        "quantum computing": ["quantum algorithms", "quantum information theory", "quantum cryptography", "quantum simulation"],
        "biotechnology": ["genetic engineering", "biopharmaceuticals", "bioinformatics", "synthetic biology", "biomaterials"],
        "climate change": ["global warming", "climate modeling", "renewable energy", "carbon capture", "climate policy"],
        "cybersecurity": ["network security", "cryptography", "information security", "cybercrime", "data privacy"],
        "economics": ["macroeconomics", "microeconomics", "behavioral economics", "econometrics", "development economics"],
        "psychology": ["cognitive psychology", "social psychology", "developmental psychology", "clinical psychology", "neuroscience"],
        "sociology": ["social inequality", "social networks", "culture", "gender studies", "race and ethnicity"],
        "history": ["ancient history", "medieval history", "modern history", "world history", "military history"],
        "literature": ["poetry", "fiction", "drama", "nonfiction", "literary theory"],
        "philosophy": ["ethics", "metaphysics", "epistemology", "logic", "political philosophy"],
        "physics": ["particle physics", "condensed matter physics", "astrophysics", "quantum mechanics", "general relativity"],
        "chemistry": ["organic chemistry", "inorganic chemistry", "analytical chemistry", "physical chemistry", "biochemistry"],
        "biology": ["cell biology", "molecular biology", "genetics", "ecology", "evolutionary biology"]
    }
    if user_input.lower() in expanded_terms:
        return expanded_terms[user_input.lower()]
    else:
        return [user_input] 

def main():
    """Main function to orchestrate the workflow."""
    api_key = SERPAPI_API_KEY  # Replace with your actual SerpApi API key
    # Simulating user input - you can get this input from a user interface
    user_interests = ["artificial intelligence"] 

    os.makedirs(TEMP_DIR, exist_ok=True)  # Create temp folder once at the beginning
    os.makedirs(EMAILS_DIR, exist_ok=True)

    for user_interest in user_interests:
        expanded_terms = expand_query(user_interest)
        for expanded_term in expanded_terms:
            search_query = expanded_term
            today = datetime.date.today()

            results = find_papers(api_key, search_query, today)

            if results:
                process_search_results(results)
            else:
                print("No results found.")

    # Optional: Delete temp folder after processing all results
    #shutil.rmtree(TEMP_DIR, ignore_errors=True) 

if __name__ == "__main__":
    main()