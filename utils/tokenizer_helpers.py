from transformers import GPT2Tokenizer, BertTokenizer

# Load tokenizers once
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def analyze_tokenization(text):
    """Tokenize text with GPT-2 and BERT, return stats."""
    if not text.strip():
        return {"error": "Empty text provided"}
    
    # GPT-2
    gpt2_tokens = gpt2_tokenizer.tokenize(text)
    gpt2_ids = gpt2_tokenizer.encode(text)

    # BERT
    bert_tokens = bert_tokenizer.tokenize(text)
    bert_ids = bert_tokenizer.encode(text)

    return {
        "gpt2": {"tokens": gpt2_tokens, "ids": gpt2_ids, "count": len(gpt2_ids)},
        "bert": {"tokens": bert_tokens, "ids": bert_ids, "count": len(bert_ids)},
    }
