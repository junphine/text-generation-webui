import datetime
from pathlib import Path
import re
import pandas as pd
import torch
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from modules import shared
from modules.models import clear_torch_cache, load_model, unload_model
from modules.models_settings import get_model_metadata, update_model_parameters
from modules.text_generation import encode


def load_past_evaluations():
    if Path('logs/evaluations.csv').exists():
        df = pd.read_csv(Path('logs/evaluations.csv'), dtype=str)
        df['Perplexity'] = pd.to_numeric(df['Perplexity'])
        return df
    else:
        return pd.DataFrame(columns=['Model', 'LoRAs', 'Dataset', 'Perplexity', 'stride', 'max_length', 'Date', 'Comment'])


past_evaluations = load_past_evaluations()


def save_past_evaluations(df):
    global past_evaluations
    past_evaluations = df
    filepath = Path('logs/evaluations.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)


def calculate_perplexity(models, input_dataset, stride, _max_length):
    '''
    Based on:
    https://huggingface.co/docs/transformers/perplexity#calculating-ppl-with-fixedlength-models
    '''

    global past_evaluations
    cumulative_log = ''
    cumulative_log += "Loading the input dataset...\n\n"
    yield cumulative_log

    # Copied from https://github.com/qwopqwop200/GPTQ-for-LLaMa/blob/triton/utils/datautils.py
    if input_dataset == 'wikitext':
        data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        text = "\n\n".join(data['text'])
    elif input_dataset == 'ptb':
        data = load_dataset('ptb_text_only', 'penn_treebank', split='validation')
        text = "\n\n".join(data['sentence'])
    elif input_dataset == 'ptb_new':
        data = load_dataset('ptb_text_only', 'penn_treebank', split='test')
        text = " ".join(data['sentence'])
    else:
        with open(Path(f'training/datasets/{input_dataset}.txt'), 'r', encoding='utf-8') as f:
            text = f.read()

    for model in models:
        if is_in_past_evaluations(model, input_dataset, stride, _max_length):
            cumulative_log += f"`{model}` has already been tested. Ignoring.\n\n"
            yield cumulative_log
            continue

        if model != 'current model':
            try:
                yield cumulative_log + f"Loading `{model}`...\n\n"
                model_settings = get_model_metadata(model)
                shared.settings.update({k: v for k, v in model_settings.items() if k in shared.settings})  # hijacking the interface defaults
                update_model_parameters(model_settings)  # hijacking the command-line arguments
                shared.model_name = model
                unload_model()
                shared.model, shared.tokenizer = load_model(shared.model_name)
            except:
                cumulative_log += f"Failed to load `{model}`. Moving on.\n\n"
                yield cumulative_log
                continue

        cumulative_log += f"Processing `{shared.model_name}`...\n\n"
        yield cumulative_log + "Tokenizing the input dataset...\n\n"
        encodings = encode(text, add_special_tokens=False)
        seq_len = encodings.shape[1]
        if _max_length:
            max_length = _max_length
        elif hasattr(shared.model.config, 'max_position_embeddings'):
            max_length = shared.model.config.max_position_embeddings
        else:
            max_length = 2048

        nlls = []
        prev_end_loc = 0
        for begin_loc in tqdm(range(0, seq_len, stride)):
            yield cumulative_log + f"Evaluating... {100*begin_loc/seq_len:.2f}%"
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encodings[:, begin_loc:end_loc]
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100
            clear_torch_cache()
            with torch.no_grad():
                outputs = shared.model(input_ids=input_ids, labels=target_ids)

                # loss is calculated using CrossEntropyLoss which averages over valid labels
                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                # to the left by 1.
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        ppl = torch.exp(torch.stack(nlls).mean())
        add_entry_to_past_evaluations(float(ppl), shared.model_name, input_dataset, stride, _max_length)
        save_past_evaluations(past_evaluations)
        cumulative_log += f"The perplexity for `{shared.model_name}` is: {float(ppl)}\n\n"
        yield cumulative_log


def add_entry_to_past_evaluations(perplexity, model, dataset, stride, max_length):
    global past_evaluations
    entry = {
        'Model': model,
        'LoRAs': ', '.join(shared.lora_names) or '-',
        'Dataset': dataset,
        'Perplexity': perplexity,
        'stride': str(stride),
        'max_length': str(max_length),
        'Date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Comment': ''
    }
    past_evaluations = pd.concat([past_evaluations, pd.DataFrame([entry])], ignore_index=True)


def is_in_past_evaluations(model, dataset, stride, max_length):
    entries = past_evaluations[(past_evaluations['Model'] == model) &
                               (past_evaluations['Dataset'] == dataset) &
                               (past_evaluations['max_length'] == str(max_length)) &
                               (past_evaluations['stride'] == str(stride))]

    if entries.shape[0] > 0:
        return True
    else:
        return False


def generate_markdown_table():
    sorted_df = past_evaluations.sort_values(by=['Dataset', 'stride', 'Perplexity', 'Date'])
    return sorted_df
	
def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, labels

def compute_accuracy_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions[0]   
    #pred_ids = logits.argmax(axis=-1)    

    c = 0.0001
    tp=0
    tn=0
    fp=0
    fn=0    
    for i in range(0,len(pred_ids)):
        line = labels_ids[i]
        pred_line = pred_ids[i]
        for j in range(2,len(line)):
            if j>3 and pred_line[j]>=32595 and pred_line[j]<32695:
                continue
            if line[j]>1:
                if line[j]==pred_line[j]:
                    tp+=1
                else:
                    fp+=1
                c+=1     
    precision = tp/(tp+fp)
    recall = tp/c
    #pred_ids = np.where(pred_ids != -100, pred_ids, shared.tokenizer.pad_token_id)
    #labels_ids = np.where(labels_ids != -100, labels_ids, shared.tokenizer.pad_token_id)

    # all unnecessary tokens are removed
    #pred_str = shared.tokenizer.batch_decode(pred_ids, skip_special_tokens=True,clean_up_tokenization_spaces=True)
    #label_str = shared.tokenizer.batch_decode(labels_ids, skip_special_tokens=True,clean_up_tokenization_spaces=True)

    #accuracy = sum([int(pred_str[i] == label_str[i]) for i in range(len(pred_str))]) / len(pred_str)

    return {"precision": precision, "recall": recall}



def compute_set_accuracy_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions[0]
    #pred_ids = logits.argmax(axis=-1)
    tp=0
    tn=0
    fp=0
    fn=0
    pred_ids = np.where(pred_ids != -100, pred_ids, shared.tokenizer.pad_token_id)
    labels_ids = np.where(labels_ids != -100, labels_ids, shared.tokenizer.pad_token_id)

    pred_ids_new = []
    label_ids_new = []
    for i in range(0,len(pred_ids)):        
        pred_line = pred_ids[i]
        label_line = labels_ids[i]
        pred_line_ids = []
        label_line_ids = []
        for j in range(1,len(pred_line)):
            if j>3 and (pred_line[j]>=32496 and pred_line[j]<=32595 or pred_line[j]<=2):
                break
            pred_line_ids.append(pred_line[j])
        pred_ids_new.append(pred_line_ids)
        for j in range(1,len(label_line)):
            if j>3 and (label_line[j]>=32496 and label_line[j]<=32595 or label_line[j]<=2):
                break
            label_line_ids.append(label_line[j])
        label_ids_new.append(label_line_ids)    

    # all unnecessary tokens are removed    
    pred_str = shared.tokenizer.batch_decode(pred_ids_new, skip_special_tokens=False,clean_up_tokenization_spaces=True)
    label_str = shared.tokenizer.batch_decode(label_ids_new, skip_special_tokens=False,clean_up_tokenization_spaces=True)
    for i in range(0,len(pred_str)):
        #preds = pred_str[i].replace('<pad>','').replace(' ','').split('[SEP]')
        preds = re.split('�|\[SEP\]|\s',pred_str[i])
        #labels = label_str[i].replace('<pad>','').replace(' ','').split('[SEP]')
        labels = re.split('�|\[SEP\]|\s',label_str[i])
        tp0,fp0,tn0,fn0 = compute_set_accuracy_f1_matric(labels,preds)
        tp+=tp0
        fp+=fp0
        tn+=tn0
        fn+=fn0

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    if tp==0:
        f1 = 0
    else:
        f1 = 2*precision*recall/(precision+recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_set_accuracy_f1_matric(labels,preds):
    tp=0
    fp=0
    tn=0
    fn=0
    for pred in preds:
        if pred=='':
            continue
        if pred in labels:
            tp+=1
        else:
            fp+=1
    for label in labels:
        if label=='':
            continue
        if label not in preds:
            fn+=1
    return (tp,fp,tn,fn)
