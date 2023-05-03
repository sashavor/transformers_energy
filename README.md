# Transformers Energy (WIP)

## Setup

Install all requirements (within a virtualenv or Conda env) with:

```shell
pip install -r requirements.txt --upgrade
```

## Running experiments

There are two configurations:
- fill-mask
- text-generation

To run an individual config on the default parameters, use:

```shell
python measure_energy_usage.py --config-name=<CONFIG NAME> > /dev/null
```

You will find the energy logs under `outputs/`. The hydra config for the run will be in the run directory, in `.hydra/`.

### Sweeps

Run _every combination_ for a particular config with:

```shell
python measure_energy_usage.py --config-name=<CONFIG NAME> hydra.mode=MULTIRUN > /dev/null
```

You will find the energy logs under `multiruns/`. The hydra config for each run will be in the run directories, in `.hydra/`.

See:
- https://hydra.cc/docs/patterns/configuring_experiments/#sweeping-over-experiments
- https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/

## PyJoules Logs

All PyJoules data will be logged to `measure_energy_usage.log` under the Hydra run, along with any other logs from the run. PyJoules data will be prefixed with `[PYJOULES] `, so any log file can be filtered into a CSV of PyJoules data with the following:

```shell
grep "\[PYJOULES\]" /PATH/TO/measure_energy_usage.log | sed -e "s/\[PYJOULES\] //g" > /PATH/TO/pyjoules_data.csv
```

## Text Generation

From: https://huggingface.co/docs/transformers/v4.28.1/en/generation_strategies#decoding-strategies

> The default decoding strategy is greedy search, which is the simplest decoding strategy that picks a token with the highest probability as the next token.

For this reason, for Text Generation we sweep over these strategies:

- Greedy Search
- Contrastive Search
- Multinomial Sampling
- Beam Search Decoding
- Beam Search Multinomial Sampling
- Diverse Beam Search Decoding

## References

Hydra

```text
@Misc{Yadan2019Hydra,
  author =       {Omry Yadan},
  title =        {Hydra - A framework for elegantly configuring complex applications},
  howpublished = {Github},
  year =         {2019},
  url =          {https://github.com/facebookresearch/hydra}
}
```