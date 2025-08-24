from google import genai
from config import API_KEY, PRICE_PER_1K_TOKENS

# Initialize Gemini client
client = genai.Client(api_key=API_KEY)

def summarize_text(text):
    """Summarize input text using Gemini LLM."""
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"Summarize the text: {text}"
        )
        summary = response.text
        tokens_used = len(text.split())  # rough token estimate
        cost = (tokens_used / 1000) * PRICE_PER_1K_TOKENS
        return summary, tokens_used, cost
    except Exception as e:
        return f"Error during summarization: {e}", 0, 0
