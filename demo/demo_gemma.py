import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def load_generator():
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.capture_scalar_outputs = False
    torch._dynamo.disable()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_enable_fp32_cpu_offload=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b-it",
        quantization_config=bnb_config,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    return model, tokenizer


def generate_article(statement, model, tokenizer):
    prompt = f"[INST] Imagine that you are a journalist working in a newspaper. Write an article for the following subject: '{statement}'. Please write it as one continuous block of text, no formatting, no captioning, no headings. [/INST]"
    inputs = tokenizer(
        prompt, return_tensors="pt", padding=True, truncation=True
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    full_output = decoded[0]
    return (
        full_output.split("[/INST]")[-1]
        .strip()
        .replace("\t", "")
        .replace("\n", "")
    )
