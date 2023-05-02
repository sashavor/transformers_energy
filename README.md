# Transformers Energy (WIP)

## Run development setup

```shell
python measure_energy_usage.py > /dev/null
```

You will find the energy logs under `outputs/`. The hydra config for the run will be in the run directory, in `.hydra/`.

## Running experiments

Run _every combination_ with:

```shell
python measure_energy_usage.py hydra.mode=MULTIRUN > /dev/null
```

You will find the energy logs under `multiruns/`. The hydra config for each run will be in the run directories, in `.hydra/`.

See:
- https://hydra.cc/docs/patterns/configuring_experiments/#sweeping-over-experiments
- https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/

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