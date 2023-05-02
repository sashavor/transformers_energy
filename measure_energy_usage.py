import hydra
from omegaconf import DictConfig

from pyJoules.energy_meter import measure_energy
from pyJoules.energy_trace import EnergyTrace
from pyJoules.handler import EnergyHandler

from transformers import logging as transfomers_logging
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm
import logging


log = logging.getLogger(__name__)
transfomers_logging.set_verbosity_error()  # To suppress warnings about model weight loading (see link below)
# https://discuss.huggingface.co/t/is-some-weights-of-the-model-were-not-used-warning-normal-when-pre-trained-bert-only-by-mlm/5672/2
api = HfApi()


class HydraHandler(EnergyHandler):
    headers_printed = False

    def process(self, trace: EnergyTrace):
        if not self.headers_printed:
            domain_names = trace[0].energy.keys()
            log.info('timestamp;tag;duration;' + ';'.join(domain_names))
            self.headers_printed = True

        for sample in trace:
            line_beginning = f'{sample.timestamp};{sample.tag};{sample.duration};'
            energy_values = [str(sample.energy[domain]) for domain in sample.energy.keys()]
            log.info(line_beginning + ';'.join(energy_values))


def setup_and_run_inference(
        model,
        tokenizer,
        task_type,
        dataset,
        sequence_length,
        device
):
    def prepare_data_items(item):
        return {
            "input_string": tokenizer.convert_tokens_to_string(tokenizer.tokenize(
                item[dataset.text_feature],
                max_length=sequence_length,
                truncation=True
            ))
        }

    def fill_mask_preprocess(item):
        return {
            "input_tensor": tokenizer(
                " ".join([item["input_string"], tokenizer.special_tokens_map["mask_token"]]),
                return_tensors="pt",
            ).to(device)
        }

    def text_generation_preprocess(item):
        return {
            "input_tensor": tokenizer(item["input_string"], return_tensors="pt").to(device)
        }

    ds = load_dataset(**dataset.load_args, streaming=True).map(prepare_data_items)

    if task_type == "fill-mask":
        ds = ds.map(fill_mask_preprocess)
    if task_type == "text-generation":
        ds = ds.map(text_generation_preprocess)
    else:
        raise NotImplementedError(f"Task type \"{task_type}\" is not supported.")

    progress_bar = tqdm(total=dataset.take)
    ds_iter = iter(ds.take(dataset.take))

    @measure_energy(handler=HydraHandler())
    def run_inference(x):
        return model(**x)  # We don't actually care about the result.

    for d in ds_iter:
        run_inference(d["input_tensor"])
        progress_bar.update(1)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_experiments(cfg: DictConfig) -> None:
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    model = AutoModel.from_pretrained(cfg.model.name).to(cfg.device)
    task_type =api.model_info(cfg.model.name).pipeline_tag

    setup_and_run_inference(
        model=model,
        tokenizer=tokenizer,
        task_type=task_type,
        dataset=cfg.dataset,
        sequence_length=cfg.sequence_length,
        device=cfg.device
    )


if __name__ == "__main__":
    run_experiments()
