import sys
import torch
import pytorch_lightning as pl
sys.path.append('donut')
from donut_model_optimized import DonutConfig, DonutModel
from sconf import Config

class DonutModelPLModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if self.config.get("pretrained_model_name_or_path", False):
            self.model = DonutModel.from_pretrained(
                self.config.pretrained_model_name_or_path,
                input_size=self.config.input_size,
                max_length=self.config.max_length,
                align_long_axis=self.config.align_long_axis,
                ignore_mismatched_sizes=True,
            )
        else:
            self.model = DonutModel(
                config=DonutConfig(
                    input_size=self.config.input_size,
                    max_length=self.config.max_length,
                    align_long_axis=self.config.align_long_axis,
                )
            )

def convert_checkpoint(checkpoint_path, saved_model_path, output_path):
    # Load the saved model with correct tokenizer
    model = DonutModel.from_pretrained(saved_model_path)
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    print("Checkpoint keys:", checkpoint.keys())
    print("State dict keys sample:", list(checkpoint['state_dict'].keys())[:10])
    # Load the state_dict for the model
    state_dict = {k[6:]: v for k, v in checkpoint['state_dict'].items() if k.startswith('model.')}
    print("Filtered state dict keys sample:", list(state_dict.keys())[:10])
    model.load_state_dict(state_dict)
    model.save_pretrained(output_path)
    model.decoder.tokenizer.save_pretrained(output_path)
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    checkpoint_path = "/workspace/result/train_custom_receipt/VISTA/epoch-epoch=17-val_metric=0.2061.ckpt"
    saved_model_path = "/workspace/result/train_custom_receipt/VISTA001"
    output_path = "/workspace/result/train_custom_receipt/VISTA_converted"
    convert_checkpoint(checkpoint_path, saved_model_path, output_path)