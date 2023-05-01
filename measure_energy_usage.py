import hydra
from omegaconf import DictConfig
from pyJoules.energy_meter import measure_energy
from pyJoules.handler.csv_handler import CSVHandler
from transformers import AutoModel, AutoTokenizer, logging
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import logging

logging.set_verbosity_error()  # To suppress warnings about model weight loading (see link below)


def prepare_data_items(row, tokenizer, sequence_length):
    return {
        "input_string": tokenizer.convert_tokens_to_string(tokenizer.tokenize(
            row["text"],
            max_length=sequence_length,
            truncation=True
        ))
    }


def setup_and_run_inference(model, dataset, sequence_length, device):
    inference_csv_handler = CSVHandler(
        f"./energy_logs/{model}_{dataset.name}_{sequence_length}_inference_energy.csv"
    )

    # TODO: Instantiate model and tokenizer only once, outside of this function.
    tokenizer = AutoTokenizer.from_pretrained(model)

    # Note on warnings:
    # https://discuss.huggingface.co/t/is-some-weights-of-the-model-were-not-used-warning-normal-when-pre-trained-bert-only-by-mlm/5672/2
    model = AutoModel.from_pretrained(model).to(device)

    ds = load_dataset(dataset.name, split=dataset.split, streaming=True).map(
        lambda x: prepare_data_items(x, tokenizer, sequence_length)
    )

    progress_bar = tqdm(total=dataset.take)
    ds_iter = iter(ds.take(dataset.take))

    @measure_energy(handler=inference_csv_handler)
    def run_inference(x):
        return model(**x)  # TODO: There are .generate options to pass in here (hydra-controlled)

    for d in ds_iter:
        inputs = tokenizer(
            d["input_string"] + " " + tokenizer.special_tokens_map["mask_token"],
            return_tensors="pt"
        )

        run_inference(inputs)

        progress_bar.update(1)

    inference_csv_handler.save_data()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_experiments(cfg: DictConfig) -> None:
    for model in cfg.models:
        for dataset in cfg.datasets:
            for sequence_length in dataset.sequence_lengths:
                setup_and_run_inference(model, dataset, sequence_length, cfg.device)


if __name__ == "__main__":
    run_experiments()
