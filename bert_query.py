from transformers import pipeline,  set_seed
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM, AutoModelForQuestionAnswering, AutoModelForTokenClassification
from datasets import load_dataset
from pyJoules.energy_meter import measure_energy
#from pyJoules.handler.csv_handler import CSVHandler
from pyJoules.device.nvidia_device import NvidiaGPUDomain

#load_csv_handler = CSVHandler('./load_model_energy.csv')
#query_csv_handler = CSVHandler('./query_model_energy.csv')

##### Dataset loading
og_dataset = load_dataset("gem", "common_gen", split = "validation")
#sent_analysis_dataset = load_dataset("imdb", split = "test")
#qa_dataset = load_dataset("squad", split = "validation")
#ner_dataset = load_dataset("conll2003", split = "validation")

### Model loading

og_bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#og_bert_model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

#sent_analysis_bert_tokenizer = AutoTokenizer.from_pretrained("fabriceyhc/bert-base-uncased-imdb")
#sent_analysis_bert_model = AutoModelForSequenceClassification.from_pretrained("fabriceyhc/bert-base-uncased-imdb")

#qa_bert_tokenizer = AutoTokenizer.from_pretrained("salti/bert-base-multilingual-cased-finetuned-squad")
#qa_bert_model = AutoModelForQuestionAnswering.from_pretrained("salti/bert-base-multilingual-cased-finetuned-squad")

#ner_bert_tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
#ner_bert_model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")


#@measure_energy(domains=[NvidiaGPUDomain(0)],handler=csv_handler)
#def query_gen_model(gen, data):
#      gen(data, max_length=50, num_return_sequences=1)

#(handler=query_csv_handler)

@measure_energy
def query_mlm_model(pipe, data):
    query = pipe(data)
    return query

#(handler=query_csv_handler)
@measure_energy
def query_sent_model(pipeline, data):
    query = pipeline(data)

#handler=query_csv_handler
@measure_energy
def query_qa_model(pipeline, data):
    query = pipeline(data)

#(handler=query_csv_handler)
@measure_energy
def query_ner_model(pipeline, data):
    query = pipeline(data)

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
