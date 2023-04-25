from transformers import pipeline
from datasets import load_dataset
from pyJoules.energy_meter import measure_energy
from pyJoules.handler.csv_handler import CSVHandler
from pyJoules.device.nvidia_device import NvidiaGPUDomain

csv_handler = CSVHandler('./energy.csv')

dataset = load_dataset("Ericwang/promptSentiment")

from transformers import pipeline, set_seed

@measure_energy(domains=[NvidiaGPUDomain(0)],handler=csv_handler)
def query_gen_model(gen, data):
      gen(data, max_length=50, num_return_sequences=1)


def main():
    generator = pipeline('text-generation', model= 'gpt2', device=0, pad_token_id = 50256)
    for i in dataset['train']['text'][:1000]:
        query_gen_model(generator, ' '.join(i.split()[:15]))

if __name__ == "__main__":
    main()
    csv_handler.save_data()
