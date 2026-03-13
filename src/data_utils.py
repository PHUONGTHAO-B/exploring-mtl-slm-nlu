from datasets import load_dataset
from transformers import DistilBertTokenizerFast, BertTokenizerFast

# -------------------------------------------------
# Choose tokenizer based on model type
# -------------------------------------------------

def get_tokenizer(model_type="slm"):
    """
    model_type:
        "slm" -> DistilBERT tokenizer
        "llm" -> BERT-base tokenizer
    """
    if model_type == "llm":
        return BertTokenizerFast.from_pretrained(
            "bert-base-uncased"
        )
    else:
        return DistilBertTokenizerFast.from_pretrained(
            "distilbert-base-uncased"
        )

# -------------------------------------------------
# Tokenization
# -------------------------------------------------

def tokenize(batch, tokenizer):

    if "sentence" in batch:
        return tokenizer(
            batch["sentence"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

    if "question1" in batch:
        return tokenizer(
            batch["question1"],
            batch["question2"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

    if "sentence1" in batch:
        return tokenizer(
            batch["sentence1"],
            batch["sentence2"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

# -------------------------------------------------
# Dataset loaders
# -------------------------------------------------

def load_sst2(model_type="slm"):
    tokenizer = get_tokenizer(model_type)
    ds = load_dataset("glue", "sst2")
    ds = ds.map(lambda x: tokenize(x, tokenizer), batched=True)
    ds.set_format(type="torch",
                  columns=["input_ids","attention_mask","label"])
    return ds

def load_qqp(model_type="slm"):
    tokenizer = get_tokenizer(model_type)
    ds = load_dataset("glue", "qqp")
    ds = ds.map(lambda x: tokenize(x, tokenizer), batched=True)
    ds.set_format(type="torch",
                  columns=["input_ids","attention_mask","label"])
    return ds

def load_stsb(model_type="slm"):
    tokenizer = get_tokenizer(model_type)
    ds = load_dataset("glue", "stsb")
    ds = ds.map(lambda x: tokenize(x, tokenizer), batched=True)
    ds.set_format(type="torch",
                  columns=["input_ids","attention_mask","label"])
    return ds
