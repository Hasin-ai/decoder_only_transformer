import os
import json
import requests
import tiktoken
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel

DATA_DIRECTORY = os.path.join(os.path.dirname(__file__), "hellaswag")

HELLASWAG_URLS = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

tokenizer = tiktoken.get_encoding("gpt2")

def download_file_from_url(url: str, destination_path: str, chunk_size=1024):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    with open(destination_path, "wb") as file_handle, tqdm(
        desc=destination_path,
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            written_size = file_handle.write(chunk)
            progress_bar.update(written_size)

def ensure_data_downloaded(split_name: str):
    os.makedirs(DATA_DIRECTORY, exist_ok=True)
    url = HELLASWAG_URLS[split_name]
    file_path = os.path.join(DATA_DIRECTORY, f"hellaswag_{split_name}.jsonl")
    if not os.path.exists(file_path):
        print(f"Downloading {url} to {file_path}...")
        download_file_from_url(url, file_path)

def prepare_example_tensors(example_dict):
    context = example_dict["ctx"]
    correct_label = example_dict["label"]
    candidate_endings = example_dict["endings"]

    example_data = {
        "label": correct_label,
        "context_token_ids": None,
        "ending_token_ids": [],
    }

    context_token_ids = tokenizer.encode(context)
    example_data["context_token_ids"] = context_token_ids

    token_sequences = []
    completion_masks = []
    for ending_text in candidate_endings:
        ending_token_ids = tokenizer.encode(" " + ending_text)
        token_sequences.append(context_token_ids + ending_token_ids)
        completion_masks.append([0] * len(context_token_ids) + [1] * len(ending_token_ids))
        example_data["ending_token_ids"].append(ending_token_ids)

    max_sequence_length = max(len(seq) for seq in token_sequences)
    tokens_tensor = torch.zeros((4, max_sequence_length), dtype=torch.long)
    mask_tensor = torch.zeros((4, max_sequence_length), dtype=torch.long)
    for idx, (seq, mask) in enumerate(zip(token_sequences, completion_masks)):
        tokens_tensor[idx, :len(seq)] = torch.tensor(seq)
        mask_tensor[idx, :len(mask)] = torch.tensor(mask)

    return example_data, tokens_tensor, mask_tensor, correct_label

def iterate_dataset_examples(split_name: str):
    ensure_data_downloaded(split_name)
    file_path = os.path.join(DATA_DIRECTORY, f"hellaswag_{split_name}.jsonl")
    with open(file_path, "r") as file:
        for line in file:
            yield json.loads(line)

@torch.no_grad()
def run_evaluation(model_name: str, device_name: str):
    torch.set_float32_matmul_precision('high')
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device_name)

    total_examples = 0
    correct_predictions_raw = 0
    correct_predictions_normalized = 0

    for example in iterate_dataset_examples("val"):
        example_data, input_tokens, completion_mask, true_label = prepare_example_tensors(example)
        input_tokens = input_tokens.to(device_name)
        completion_mask = completion_mask.to(device_name)

        logits = model(input_tokens).logits
        shifted_logits = logits[..., :-1, :].contiguous()
        shifted_tokens = input_tokens[..., 1:].contiguous()

        flat_logits = shifted_logits.view(-1, shifted_logits.size(-1))
        flat_tokens = shifted_tokens.view(-1)

        losses = F.cross_entropy(flat_logits, flat_tokens, reduction='none')
        losses = losses.view(input_tokens.size(0), -1)

        shifted_mask = completion_mask[..., 1:].contiguous()
        masked_losses = losses * shifted_mask

        sum_losses = masked_losses.sum(dim=1)
        avg_losses = sum_losses / shifted_mask.sum(dim=1)

        predicted_index_raw = sum_losses.argmin().item()
        predicted_index_norm = avg_losses.argmin().item()

        total_examples += 1
        correct_predictions_raw += int(predicted_index_raw == true_label)
        correct_predictions_normalized += int(predicted_index_norm == true_label)

        print(f"{total_examples} acc_norm: {correct_predictions_normalized}/{total_examples} = {correct_predictions_normalized/total_examples:.4f}")

        if total_examples <= 10:
            print("---")
            print(f"Context:\n {example['ctx']}")
            print("Candidate endings:")
            for idx, ending_text in enumerate(example["endings"]):
                print(f"{idx} (loss: {avg_losses[idx].item():.4f}) {ending_text}")
            print(f"Predicted: {predicted_index_norm}, Actual: {true_label}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", type=str, default="gpt2", help="Pretrained model name or path")
    parser.add_argument("-d", "--device_name", type=str, default="cuda", help="Computation device to use")
    args = parser.parse_args()
    run_evaluation(args.model_name, args.device_name)
