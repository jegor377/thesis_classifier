import torch
import gradio as gr
import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel
from demo_utils import (
    statement_to_num_metadata,
    speaker_to_num_metadata,
    num_metadata_cols,
    one_hot_cols,
    one_hot_labels,
    label_to_index,
    one_hot_metadata_size,
    limit_tokens,
    LiarPlusSingleFinetunedRoBERTasClassifier,
    examples,
    states,
    ids2labels
)
from demo_pipelines import load_pipelines, run_pipeline
from demo_gemma import generate_article, load_generator


# MODEL LOADING
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
roberta = RobertaModel.from_pretrained("roberta-base")

model = LiarPlusSingleFinetunedRoBERTasClassifier(
    encoder_model=roberta,
    num_metadata_len=len(num_metadata_cols),
    one_hot_metadata_size=one_hot_metadata_size,
    num_hidden=128,
    num_classes=6,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.load_state_from_save(torch.load("best_model.pth")["model_state_dict"])
model.eval()


# INPUT BUILDERS
def build_string_input(
    statement,
    subject,
    speaker,
    job_title,
    state,
    party_affiliation,
    context,
    tokenizer,
    gemma_model,
    gemma_tokenizer,
):
    return (
        f"{limit_tokens(tokenizer, f'[STATEMENT] {statement}', 128)} "
        f"{limit_tokens(tokenizer, f'[ARTICLE] {generate_article(statement, gemma_model, gemma_tokenizer)}', 128)} "
        f"{limit_tokens(tokenizer, f'[SUBJECT] {subject}', 15)} "
        f"{limit_tokens(tokenizer, f'[SPEAKER] {speaker}', 15)} "
        f"{limit_tokens(tokenizer, f'[JOB_TITLE] {job_title}', 15)} "
        f"{limit_tokens(tokenizer, f'[STATE] {state}')} "
        f"{limit_tokens(tokenizer, f'[PARTY_AFFILIATION] {party_affiliation}')} "
        f"{limit_tokens(tokenizer, f'[CONTEXT] {context}', 15)}"
    )


def build_num_input(statement, speaker):
    if statement in statement_to_num_metadata:
        return statement_to_num_metadata[statement]

    if speaker in speaker_to_num_metadata:
        val = speaker_to_num_metadata[speaker].copy()
        val[-1] = 0.0
        val[-2] = 0.0
        return val

    return [0.0 for _ in num_metadata_cols]


def build_one_hot_input(statement, pipelines):
    one_hot_metadata_list = []
    for column in one_hot_cols:
        value = run_pipeline(column, statement, pipelines)
        if value == "noise":
            value = "word salad"
        idx = label_to_index[column][value]
        one_hot = F.one_hot(torch.tensor(idx), num_classes=len(one_hot_labels[column]))
        one_hot_metadata_list.append(one_hot)

    return one_hot_metadata_list


# PREDICTION FUNCTION
def prediction(
    statement,
    subject,
    speaker,
    job_title,
    state,
    party_affiliation,
    context,
    pipelines,
    gemma_model,
    gemma_tokenizer,
):
    input_text = build_string_input(
        statement,
        subject,
        speaker,
        job_title,
        state,
        party_affiliation,
        context,
        tokenizer,
        gemma_model,
        gemma_tokenizer,
    )

    encoded = tokenizer(
        input_text,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    num_metadata = (
        torch.tensor(build_num_input(statement, speaker), dtype=torch.float)
        .unsqueeze(0)
        .to(device)
    )

    one_hot_metadata_list = build_one_hot_input(statement, pipelines)

    one_hot_metadata = (
        torch.cat(one_hot_metadata_list, dim=0)
        .unsqueeze(0)
        .float()
        .to(device)
    )

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_metadata=num_metadata,
            one_hot_metadata=one_hot_metadata,
        )
        predicted_idx = torch.argmax(outputs, dim=1).item()

    return ids2labels[predicted_idx]



# GRADIO APP
def main():
    pipelines = load_pipelines()
    gemma_model, gemma_tokenizer = load_generator()

    demo = gr.Interface(
        fn=lambda *args: prediction(
            *args, pipelines, gemma_model, gemma_tokenizer
        ),
        inputs=[
            gr.Textbox(label="Statement", max_length=300),
            gr.Textbox(label="Subject", max_length=50),
            gr.Textbox(label="Speaker", max_length=50),
            gr.Textbox(label="Job title", max_length=50),
            gr.Dropdown(states, label="State"),
            gr.Textbox(label="Party affiliation", max_length=50),
            gr.Textbox(label="Context", max_length=50),
        ],
        outputs=gr.Label(label="Wynik klasyfikacji"),
        examples=examples,
        allow_flagging="never"
    )

    demo.launch()


if __name__ == "__main__":
    main()