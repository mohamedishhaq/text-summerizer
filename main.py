import gradio as gr
from utils.llm_helpers import summarize_text
from utils.tokenizer_helpers import analyze_tokenization
from langdetect import detect
from textblob import TextBlob


def analyze_text(user_input: str):
    if not user_input.strip():
        return (
            "❌ No text provided.",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    # --- Summarization ---
    summary, tokens_used, cost = summarize_text(user_input)

    # --- Tokenization ---
    token_stats = analyze_tokenization(user_input)
    gpt2_tokens, gpt2_count, bert_tokens, bert_count = None, None, None, None
    if "error" not in token_stats:
        gpt2_tokens = token_stats["gpt2"]["tokens"]
        gpt2_count = token_stats["gpt2"]["count"]
        bert_tokens = token_stats["bert"]["tokens"]
        bert_count = token_stats["bert"]["count"]

    # --- Language Detection ---
    lang = detect(user_input)

    # --- Sentiment Analysis ---
    sentiment = TextBlob(user_input).sentiment

    return (
        summary,
        tokens_used,
        f"${cost:.6f}",
        gpt2_tokens,
        gpt2_count,
        bert_tokens,
        bert_count,
        f"Polarity={sentiment.polarity:.2f}, Subjectivity={sentiment.subjectivity:.2f}",
        lang,
    )


with gr.Blocks() as demo:
    gr.Markdown("## 📝 Text Analysis Tool")

    with gr.Row():
        text_input = gr.Textbox(
            label="Enter text",
            placeholder="Type or paste your text here...",
            lines=6,
        )

    submit_btn = gr.Button("Analyze")

    with gr.Tab("Summary"):
        summary_output = gr.Textbox(label="Gemini Summary")
        tokens_used_output = gr.Number(label="Tokens Used (Estimate)")
        cost_output = gr.Textbox(label="Cost ($)")

    with gr.Tab("Tokenization"):
        gpt2_tokens_output = gr.Textbox(label="GPT-2 Tokens")
        gpt2_count_output = gr.Number(label="GPT-2 Count")
        bert_tokens_output = gr.Textbox(label="BERT Tokens")
        bert_count_output = gr.Number(label="BERT Count")

    with gr.Tab("Language & Sentiment"):
        lang_output = gr.Textbox(label="Detected Language")
        sentiment_output = gr.Textbox(label="Sentiment Analysis")

    submit_btn.click(
        fn=analyze_text,
        inputs=text_input,
        outputs=[
            summary_output,
            tokens_used_output,
            cost_output,
            gpt2_tokens_output,
            gpt2_count_output,
            bert_tokens_output,
            bert_count_output,
            sentiment_output,
            lang_output,
        ],
    )

if __name__ == "__main__":
    demo.launch()
