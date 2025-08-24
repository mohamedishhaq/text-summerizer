from utils.llm_helpers import summarize_text
from utils.tokenizer_helpers import analyze_tokenization
from langdetect import detect
from textblob import TextBlob

def main():
    user_input = input("Enter text to analyze:\n> ")

    if not user_input.strip():
        print("❌ No text provided. Please try again.")
        return

    # --- Summarization ---
    summary, tokens_used, cost = summarize_text(user_input)
    print("\n🔹 Gemini Summary:")
    print(summary)
    print(f"📊 Tokens used (estimate): {tokens_used} | 💰 Cost: ${cost:.6f}")
    print("=" * 60)

    # --- Tokenization Analysis ---
    print("\n🔹 Tokenization Analysis (GPT-2 vs BERT)")
    token_stats = analyze_tokenization(user_input)
    if "error" in token_stats:
        print(token_stats["error"])
    else:
        print("\n🟢 GPT-2:")
        print("Tokens:", token_stats["gpt2"]["tokens"])
        print("Count:", token_stats["gpt2"]["count"])

        print("\n🔵 BERT:")
        print("Tokens:", token_stats["bert"]["tokens"])
        print("Count:", token_stats["bert"]["count"])

    print("=" * 60)

    # --- Advanced Feature 1: Language Detection ---
    lang = detect(user_input)
    print(f"\n🌍 Language Detected: {lang}")

    # --- Advanced Feature 2: Sentiment Analysis ---
    sentiment = TextBlob(user_input).sentiment
    print(f"\n💡 Sentiment Analysis: Polarity={sentiment.polarity:.2f}, Subjectivity={sentiment.subjectivity:.2f}")

    print("=" * 60)

if __name__ == "__main__":
    main()
