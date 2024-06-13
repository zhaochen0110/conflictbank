import json
import os
import argparse
from tqdm import tqdm
import torch
from transformers import pipeline, AutoModel, AutoTokenizer
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split
import random
import pyarrow as pa
from nltk.tokenize import sent_tokenize
import nltk
import glob

def truncate_incomplete_sentence(evidence):
    """
    Truncate incomplete sentences from the evidence text.
    :param evidence: str, the evidence text
    :return: str, the truncated evidence text
    """
    sentences = sent_tokenize(evidence)
    complete_sentences = sentences[:-1] if len(sentences) > 1 else sentences
    return ' '.join(complete_sentences)

def read_arrow_to_df_julia_ok(path):
    """
    Read a PyArrow file and convert it to a Pandas DataFrame.
    :param path: str, path to the PyArrow file
    :return: DataFrame, the resulting DataFrame
    """
    with open(path, "rb") as f:
        r = pa.ipc.RecordBatchStreamReader(f)
        df = r.read_pandas()
        return df

class NLI_classifier(nn.Module):
    """
    Neural Network for Natural Language Inference (NLI) classification.
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 3)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
def mean_pooling(model_output, attention_mask):
    """
    Mean pooling function to aggregate token embeddings.
    :param model_output: tensor, model output
    :param attention_mask: tensor, attention mask
    :return: tensor, aggregated embeddings
    """
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def embedding(tokenizer, embedding_model, sentences):
    """
    Generate embeddings for sentences using a pre-trained model.
    :param tokenizer: tokenizer, the tokenizer
    :param embedding_model: model, the embedding model
    :param sentences: list, list of sentences
    :return: tensor, the concatenated vector
    """
    encoded_input = tokenizer(sentences, padding="max_length", truncation=True, return_tensors='pt', max_length=512).to('cuda')
    model_output = embedding_model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    u = sentence_embeddings[0]
    sss = sentence_embeddings[1]
    difference = torch.abs(u - sss)
    concatenated_vector = difference
    return concatenated_vector

def nli_classifier(classifier_model, concatenated_vector):
    """
    Classify the NLI task using the classifier model.
    :param classifier_model: model, the classifier model
    :param concatenated_vector: tensor, the input vector
    :return: int, the predicted class
    """
    out = classifier_model(concatenated_vector)
    _, predicted_class = torch.max(out, 0)
    return predicted_class

def main(raw_file, nli_model, embedding_model, classifier_model, selected_raw_data_path, question_template_path, output_path, relation_to_object, qid_names, batch_size):
    """
    Main function to process raw data and generate conflict data.
    """
    # Initialize the text classification pipeline
    pipe = pipeline("text-classification", model=nli_model, device=0, batch_size=batch_size)

    # Read raw data
    raw_datas = []
    with open(raw_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            raw_datas.append(data)

    selected_data = []
    for i in tqdm(range(0, len(raw_datas), batch_size)):
        correct_result = pipe([{'text': raw_datas[j]['correct_pair'], 'text_pair': raw_datas[j]['correct_evidence']} for j in range(i, min(len(raw_datas), i + batch_size))], padding=True)

        misinformation, temporal, semantic = [], [], []
        misinformation_cnt, temporal_cnt, semantic_cnt = {}, {}, {}
        misinformation_ind, temporal_ind, semantic_ind = 0, 0, 0

        for j in range(i, min(len(raw_datas), i + batch_size)):
            data = raw_datas[j]
            if not (len(data['fact_conflict_evidence']) < 50 or 'I apologize' in data['fact_conflict_evidence'] or 'I cannot' in data['fact_conflict_evidence']):
                misinformation.append({'text': data['fact_conflict_evidence'], 'text_pair': data['fact_conflict_evidence']})
                misinformation_cnt[data['fact_conflict_evidence']] = misinformation_ind
                misinformation_ind += 1
            if not (len(data['temporal_conflict_evidence']) < 50 or 'I apologize' in data['temporal_conflict_evidence'] or 'I cannot' in data['temporal_conflict_evidence']):
                temporal.append({'text': data['temporal_conflict_evidence'], 'text_pair': data['temporal_conflict_evidence']})
                temporal_cnt[data['temporal_conflict_evidence']] = temporal_ind
                temporal_ind += 1
            if not (len(data['semantic_conflict_evidence']) < 50 or 'I apologize' in data['semantic_conflict_evidence'] or 'I cannot' in data['semantic_conflict_evidence']):
                semantic.append({'text': data['semantic_conflict_evidence'], 'text_pair': data['semantic_conflict_evidence']})
                semantic_cnt[data['semantic_conflict_evidence']] = semantic_ind
                semantic_ind += 1

        misinformation_result = pipe(misinformation, padding=True)
        temporal_result = pipe(temporal, padding=True)
        semantic_result = pipe(semantic, padding=True)

        for j in range(i, min(len(raw_datas), i + batch_size)):
            data = raw_datas[j]
            if correct_result[j - i]['label'] != 'entailment':
                continue
            if data["fact_conflict_evidence"] not in misinformation_cnt or misinformation_result[misinformation_cnt[data['fact_conflict_evidence']]]['label'] != 'entailment':
                continue
            if data["temporal_conflict_evidence"] not in temporal_cnt or temporal_result[temporal_cnt[data['temporal_conflict_evidence']]]['label'] != 'entailment':
                continue
            if data["semantic_conflict_evidence"] not in semantic_cnt or semantic_result[semantic_cnt[data['semantic_conflict_evidence']]]['label'] != 'entailment':
                continue
            selected_data.append(data)

    with open('nli_result.json', 'w', encoding='utf-8') as f:
        for data in selected_data:
            json_data = json.dumps(data)
            f.write(json_data + '\n')

    tokenizer = AutoTokenizer.from_pretrained(embedding_model)
    embedding_model = AutoModel.from_pretrained(embedding_model).to('cuda')
    classifier_model = torch.load(classifier_model, map_location='cuda:0')
    classifier_model.eval()

    with open(selected_raw_data_path, 'w', encoding='utf-8') as f:
        for data in tqdm(selected_data):
            flag = True
            for conflict_type in ['fact_conflict_evidence', 'temporal_conflict_evidence', 'semantic_conflict_evidence']:
                if conflict_type in data:
                    sentences = [data['correct_evidence'], data[conflict_type]]
                    sentences_vector = embedding(tokenizer, embedding_model, sentences)
                    clas = nli_classifier(classifier_model, sentences_vector)
                    if clas != 2:
                        flag = False
                        break
            if flag:
                json_data = json.dumps(data)
                f.write(json_data + '\n')

    selected_data = []
    with open(selected_raw_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            selected_data.append(data)

    question_templates = {}
    with open(question_template_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            question_templates[data['relation_id']] = data['question_template']

    all_data = []
    for data in selected_data:
        question_template = question_templates[data['relation']]
        if '<subject>' not in question_template or '<object>' in question_template:
            continue
        question = question_template.replace('<subject>', data['subject'])
        data['question'] = question
        all_data.append(data)

    relation_to_object_dict = {}
    with open(relation_to_object, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            relation_to_object_dict[data['key']] = data['value']

    id_to_name = {}
    with open(qid_names, "r") as file:
        lines = file.readlines()
        for line in tqdm(lines):
            items = line.strip().split()
            if 'Q' not in items[0]:
                continue
            id_to_name[items[0]] = ' '.join(items[1:])

    with open(output_path, 'w', encoding='utf-8') as f:
        for data in tqdm(all_data):
            relation = data['relation']
            options = [data['object'], data['replaced_object']]
            cnt = 5000
            while cnt > 0 and len(options) < 4:
                add_object = random.sample(relation_to_object_dict[relation], 1)[0]
                if id_to_name[add_object] not in options:
                    options.append(id_to_name[add_object])
                cnt -= 1
            if cnt == 0:
                continue
            random.shuffle(options)
            to_options = ['A', 'B', 'C', 'D']
            correct_ind = options.index(data['object'])
            replaced_ind = options.index(data['replaced_object'])
            correct_option = to_options[correct_ind]
            replaced_option = to_options[replaced_ind]
            data['options'] = options
            data['correct_option'] = correct_option
            data['replaced_option'] = replaced_option
            json_data = json.dumps(data)
            f.write(json_data + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process raw data and generate conflict data.")
    parser.add_argument('--raw_file', required=True, help="Path to the raw data file.")
    parser.add_argument('--nli_model', required=True, help="Path to the NLI model.")
    parser.add_argument('--embedding_model', required=True, help="Path to the embedding model.")
    parser.add_argument('--classifier_model', required=True, help="Path to the classifier model.")
    parser.add_argument('--selected_raw_data_path', required=True, help="Path to the selected raw data output file.")
    parser.add_argument('--question_template_path', required=True, help="Path to the question template file.")
    parser.add_argument('--output_path', required=True, help="Path to the final output file.")
    parser.add_argument('--relation_to_object', required=True, help="Path to the relation to object file.")
    parser.add_argument('--qid_names', required=True, help="Path to the QID names file.")
    parser.add_argument('--batch_size', type=int, required=True, help="Batch size for processing.")
    args = parser.parse_args()

    main(args.raw_file, args.nli_model, args.embedding_model, args.classifier_model, args.selected_raw_data_path, args.question_template_path, args.output_path, args.relation_to_object, args.qid_names, args.batch_size)
