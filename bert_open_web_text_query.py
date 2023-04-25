from transformers import pipeline,  set_seed
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM, AutoModelForQuestionAnswering, AutoModelForTokenClassification
from datasets import load_dataset
from pyJoules.energy_meter import measure_energy
#from pyJoules.handler.csv_handler import CSVHandler
from pyJoules.device.nvidia_device import NvidiaGPUDomain

load_csv_handler = CSVHandler('./load_model_energy.csv')
query_csv_handler = CSVHandler('./query_model_energy.csv')

##### Dataset loading
og_dataset = load_dataset("gem", "common_gen", split = "validation")

og_bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

@measure_energy
def query_mlm_model(pipe, data):
    query = pipe(data)
    return query

#(handler=load_csv_handler)
@measure_energy
def load_pipeline(model,tokenizer, task, top_k):
    nlp = pipeline(task= task, top_k=top_k, model=model, tokenizer= tokenizer, device=0)
    return nlp

def main():
    unmasker = load_pipeline(task= 'fill-mask', top_k=1, model="bert-base-uncased", tokenizer= "bert-base-uncased")
    for i in og_dataset:
        start = ' '.join(i['target'].split()[:-1])
        print(start)
        print(len(og_bert_tokenizer(start)['input_ids']))
        masked = start + ' [MASK].'
        print(query_mlm_model(unmasker, masked)[0]['token_str'])

if __name__ == "__main__":
    main()
#    load_csv_handler.save_data()
#    query_csv_handler.save_data()
