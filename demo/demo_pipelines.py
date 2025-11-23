from transformers import pipeline

MODEL_CONFIGS = {
    "sentiment": {
        "task": "sentiment-analysis",
        "model": "cardiffnlp/twitter-roberta-base-sentiment",
        "post_init": lambda pipe: setattr(
            pipe.model.config,
            "id2label",
            {0: "negative", 1: "neutral", 2: "positive"},
        ),
    },
    "political_bias": {
        "task": "text-classification",
        "model": "bucketresearch/politicalBiasBERT",
    },
    "offensiveness": {
        "task": "text-classification",
        "model": "cardiffnlp/twitter-roberta-base-offensive",
    },
    "gibberish": {
        "task": "text-classification",
        "model": "madhurjindal/autonlp-Gibberish-Detector-492513457",
    },
    "emotion": {
        "task": "text-classification",
        "model": "j-hartmann/emotion-english-distilroberta-base",
    },
    "curse": {
        "task": "text-classification",
        "model": "djsull/curse_classification",
        "post_init": lambda pipe: setattr(
            pipe.model.config, "id2label", {0: "non-curse", 1: "curse"}
        ),
    },
    "question": {
        "task": "text-classification",
        "model": "mrsinghania/asr-question-detection",
        "post_init": lambda pipe: setattr(
            pipe.model.config, "id2label", {0: "not_question", 1: "question"}
        ),
    },
}


def load_pipelines():
    pipelines = {}
    for key, cfg in MODEL_CONFIGS.items():
        pipe = pipeline(
            cfg["task"], model=cfg["model"], truncation=True, max_length=512
        )
        if "post_init" in cfg:
            cfg["post_init"](pipe)
        pipelines[key] = pipe
    return pipelines


def run_pipeline(name, text, pipelines):
    try:
        return pipelines[name](text)[0]["label"]
    except Exception as e:
        return f"Error: {e}"
