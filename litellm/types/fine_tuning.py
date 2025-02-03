from openai.types.fine_tuning.fine_tuning_job import Hyperparameters


class OpenAIFineTuningHyperparameters(Hyperparameters):
    model_config = {"extra": "allow"}
