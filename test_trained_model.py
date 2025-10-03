import json
import os
import sys
from PIL import Image
import torch
import pytorch_lightning as pl
sys.path.append('donut')
from donut_model_optimized import DonutConfig, DonutModel
from donut import JSONParseEvaluator
from nltk import edit_distance
import re
from sconf import Config
class ModelOutput:
    def __init__(self, last_hidden_state, attentions=None):
        self.last_hidden_state = last_hidden_state
        self.attentions = attentions

def custom_inference(model, image, prompt, image_tensors=None, prompt_tensors=None, return_json=True, return_attentions=False):
    # prepare backbone inputs (image and prompt)
    if image is None and image_tensors is None:
        raise ValueError("Expected either image or image_tensors")
    if all(v is None for v in {prompt, prompt_tensors}):
        raise ValueError("Expected either prompt or prompt_tensors")

    if image_tensors is None:
        image_tensors = model.encoder.prepare_input(image).unsqueeze(0)

    image_tensors = image_tensors.to("cpu")

    if prompt_tensors is None:
        prompt_tensors = model.decoder.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]

    prompt_tensors = prompt_tensors.to("cpu")

    last_hidden_state = model.encoder(image_tensors)
    last_hidden_state = last_hidden_state.view(last_hidden_state.size(0), -1, last_hidden_state.size(-1))
    last_hidden_state = model.encoder_proj(last_hidden_state)

    encoder_outputs = ModelOutput(last_hidden_state=last_hidden_state, attentions=None)

    if len(encoder_outputs.last_hidden_state.size()) == 1:
        encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state.unsqueeze(0)
    if len(prompt_tensors.size()) == 1:
        prompt_tensors = prompt_tensors.unsqueeze(0)

    # get decoder output
    decoder_output = model.decoder.model.generate(
        decoder_input_ids=prompt_tensors,
        encoder_outputs=encoder_outputs,
        max_length=model.config.max_length,
        early_stopping=True,
        pad_token_id=model.decoder.tokenizer.pad_token_id,
        eos_token_id=model.decoder.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[model.decoder.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
        output_attentions=return_attentions,
    )

    output = {"predictions": list()}
    for seq in model.decoder.tokenizer.batch_decode(decoder_output.sequences):
        seq = seq.replace(model.decoder.tokenizer.eos_token, "").replace(model.decoder.tokenizer.pad_token, "")
        seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
        if return_json:
            output["predictions"].append(model.token2json(seq))
        else:
            output["predictions"].append(seq)

    if return_attentions:
        output["attentions"] = {
            "self_attentions": decoder_output.decoder_attentions,
            "cross_attentions": decoder_output.cross_attentions,
        }

    return output

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

def test_trained_model(checkpoint_path, test_data_path, test_images_path):
    # Load the trained model from the saved directory
    model = DonutModel.from_pretrained(checkpoint_path, local_files_only=True)

    model.inference = lambda image, prompt: custom_inference(model, image, prompt)

    model.eval()

    # Load test data
    with open(test_data_path, "r") as f:
        lines = f.readlines()

    test_data = [json.loads(line) for line in lines]

    predictions = []
    ground_truths = []
    scores = []

    evaluator = JSONParseEvaluator()

    for sample in test_data:
        image_path = os.path.join(test_images_path, sample["file_name"])
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        image = Image.open(image_path)

        # Run inference
        output = model.inference(image=image, prompt="<s_receipt_parsing>")["predictions"][0]

        gt = json.loads(sample["ground_truth"])["gt_parse"]

        # Calculate score using edit distance on JSON strings
        pred_str = json.dumps(output, sort_keys=True)
        gt_str = json.dumps(gt, sort_keys=True)
        score = edit_distance(pred_str, gt_str) / max(len(pred_str), len(gt_str))

        scores.append(score)

        predictions.append(output)
        ground_truths.append(gt)

        print(f"File: {sample['file_name']}")
        print(f"Prediction: {output}")
        print(f"Ground Truth: {gt}")
        print(f"Normalized Edit Distance: {score}")
        print("-" * 50)

    # Calculate overall metrics
    avg_score = sum(scores) / len(scores) if scores else 0
    ted_accuracy = 1 - avg_score  # Assuming lower edit distance is better
    f1_accuracy = evaluator.cal_f1(predictions, ground_truths)

    print(f"\nTotal samples: {len(scores)}")
    print(f"Average Normalized Edit Distance: {avg_score}")
    print(f"TED-based Accuracy: {ted_accuracy}")
    print(f"F1 Accuracy: {f1_accuracy}")

if __name__ == "__main__":
    checkpoint_path = "/workspace/result/train_custom_receipt/VISTA_converted"
    test_data_path = "donut_test/test/metadata.jsonl"
    test_images_path = "donut_test/test"

    test_trained_model(checkpoint_path, test_data_path, test_images_path)