import lightning as L
import torch
from azureml.acft.image.components.olympus.core import OlympusLightningModule
from azureml.acft.image.components.olympus.loaders import ModelCheckpointLoaderBase

def update_key_names(preloaded_state_dict):
    updated_state_dict = {}
    for key, value in preloaded_state_dict.items():
        # Replace 'lang_encoder.lang_encoder' with 'language_encoder.encoder_transformer'
        new_key = key.replace(
            "lang_encoder.lang_encoder", "language_encoder.encoder_transformer"
        )
        # Replace 'lang_encoder.logit_scale' with 'language_encoder.logit_scale'
        new_key = new_key.replace(
            "lang_encoder.logit_scale", "language_encoder.logit_scale"
        )
        # Replace 'lang_encoder.lang_proj' with 'language_encoder.lang_proj'
        new_key = new_key.replace(
            "lang_encoder.lang_proj", "language_encoder.lang_proj"
        )
        # Store the new key-value pair
        updated_state_dict["model." + new_key] = value
    return updated_state_dict


def load_orig_seem_weights(
    model, checkpoint_path: str, strict: bool = False, assign: bool = False
):
    orig_model_weights = torch.load(checkpoint_path)
    updated_weights = update_key_names(orig_model_weights)
    model_state_dict = model.state_dict()
    model_keys = set(model_state_dict.keys())
    updated_keys = set(updated_weights.keys())
    missing_keys = model_keys - updated_keys
    for key in missing_keys:
        print(f"Missing key: {key}")

    model.load_state_dict(updated_weights, strict=strict)
    print("Loaded SEEM weights")


class SEEMCheckpointLoader(ModelCheckpointLoaderBase):
    """Load old llava-rad LoRA weights directly with inline name conversion"""

    def __init__(self, strict=False, assign=False, checkpoint_path=None):
        super().__init__(strict=strict, assign=assign, checkpoint_path=checkpoint_path)

    def load(self, model: OlympusLightningModule, trainer: L.Trainer):
        print("Loading SEEM checkpoint")
        load_orig_seem_weights(
            model=model,
            checkpoint_path=self.checkpoint_path,
            strict=self.strict,
            assign=self.assign,
        )
