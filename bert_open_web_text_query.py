from transformers import pipeline,  set_seed
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM, AutoModelForQuestionAnswering, AutoModelForTokenClassification
from datasets import load_dataset
from pyJoules.energy_meter import measure_energy
from pyJoules.handler.csv_handler import CSVHandler
from pyJoules.device.nvidia_device import NvidiaGPUDomain

import subprocess


load_csv_handler = CSVHandler('./load_model_energy.csv')
query_csv_handler = CSVHandler('./query_model_energy.csv')

bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

##### Dataset loading
webtext = load_dataset("openwebtext", split= 'train', streaming=True)

@measure_energy(handler=query_csv_handler)
def query_mlm_model(pipe, data):
    query = pipe(data)
    return query


@measure_energy(handler=load_csv_handler)
def load_pipeline(model,tokenizer, task, top_k):
    nlp = pipeline(task= task, top_k=top_k, model=model, tokenizer= tokenizer, device=0)
    return nlp

def main():
    unmasker = load_pipeline(task= 'fill-mask', top_k=1, model="bert-base-uncased", tokenizer= "bert-base-uncased")
    for i in range(1,512):
            input_batch = []
            for d in iter(webtext.take(10000)):
                encoded = bert_tokenizer.tokenize(d['text'])
                truncated = encoded[:i]
                masked = ' '.join(truncated) + ' [MASK].'
                input_batch.append(masked)
            query_mlm_model(unmasker, input_batch)


if __name__ == "__main__":
    p=subprocess.call("./nvmodelprofile.sh > out", shell=True)
    main()
    load_csv_handler.save_data()
    query_csv_handler.save_data()


