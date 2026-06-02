import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

def test_model_loading():
    model = GPTNeoForCausalLM.from_pretrained("path/to/your/model")
    assert model is not None

def test_model_prediction():
    model = GPTNeoForCausalLM.from_pretrained("path/to/your/model")
    tokenizer = GPT2Tokenizer.from_pretrained("path/to/your/model")
    input_text = "Hello, world!"
    inputs = tokenizer(input_text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    assert outputs is not None