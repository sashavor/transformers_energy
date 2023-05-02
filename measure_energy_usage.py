import hydra
from omegaconf import DictConfig

from functools import reduce
from operator import add
from pyJoules.energy_meter import measure_energy
from pyJoules.energy_trace import EnergyTrace
from pyJoules.handler import EnergyHandler

from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from transformers import logging as transfomers_logging
import logging


log = logging.getLogger(__name__)
transfomers_logging.set_verbosity_error()  # To suppress warnings about model weight loading (see link below)
# https://discuss.huggingface.co/t/is-some-weights-of-the-model-were-not-used-warning-normal-when-pre-trained-bert-only-by-mlm/5672/2


class HydraHandler(EnergyHandler):
    def process(self, trace: EnergyTrace):
        for sample in trace:
            begin_string = f"start timestamp : {sample.timestamp}; tag : {sample.tag}; duration : {sample.duration}"
            energy_strings = [f"; {domain} : {value}" for domain, value in sample.energy.items()]
            log.info(f"-- ENERGY READING -- {reduce(add, energy_strings, begin_string)}")


def prepare_data_items(item, tokenizer, sequence_length):
    return {
        "input_string": tokenizer.convert_tokens_to_string(tokenizer.tokenize(
            item,
            max_length=sequence_length,
            truncation=True
        ))
    }


def setup_and_run_inference(
        model,
        tokenizer,
        dataset,
        sequence_length,
        device
):

    ds = load_dataset(dataset.name, split=dataset.split, streaming=True).map(
        lambda x: prepare_data_items(
            x[dataset.text_feature],
            tokenizer,
            sequence_length
        )
    )

    progress_bar = tqdm(total=dataset.take)
    ds_iter = iter(ds.take(dataset.take))

    @measure_energy(handler=HydraHandler())
    def run_inference(x):
        return model(**x)  # We don't actually care about the result.

    for d in ds_iter:
        inputs = tokenizer(
            " ".join([d["input_string"], tokenizer.special_tokens_map["mask_token"]]),
            return_tensors="pt",
        ).to(device)

        run_inference(inputs)

        progress_bar.update(1)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_experiments(cfg: DictConfig) -> None:
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    model = AutoModel.from_pretrained(cfg.model.name).to(cfg.device)

    setup_and_run_inference(
        model=model,
        tokenizer=tokenizer,
        dataset=cfg.dataset,
        sequence_length=cfg.sequence_length,
        device=cfg.device
    )


if __name__ == "__main__":
    run_experiments()
