import json
import argparse
def main(in_file, out_dir):
    all_datas = []
    with open(in_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            all_datas.append(data)

    ### default
    out_file = out_dir+'/'+'default.json'
    prompt_text = """According to your knowledge, choose the best choice from the following options."""
    with open(out_file, 'w', encoding='utf-8') as f:
        for data in all_datas:
            prompt = prompt_text + '\n\nQuestion: {0}\nA. {1}\nB. {2}\nC. {3}\nD. {4}\nAnswer:'.format(data['question'], data['options'][0], data['options'][1], data['options'][2], data['options'][3])
            data = {'prompt': prompt,
                    'true label': data['correct_option'],
                    'replaced label': data['replaced_option']}
            json_data = json.dumps(data)
            f.write(json_data+'\n')

    ### correct
    prompt_text = """According to the evidence provided and your knowledge, choose the best choice from the following options."""
    out_file = out_dir+'/'+'correct.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        for data in all_datas:
            prompt = prompt_text + '\n\nEvidence: {0}\nQuestion: {1}\nA. {2}\nB. {3}\nC. {4}\nD. {5}\nAnswer:'.format(data['correct_evidence'], data['question'], data['options'][0], data['options'][1], data['options'][2], data['options'][3])
            data = {'prompt': prompt,
                    'true label': data['correct_option'],
                    'replaced label': data['replaced_option']}
            json_data = json.dumps(data)
            f.write(json_data+'\n')

    ### temporal
    out_file = out_dir+'/'+'temporal.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        for data in all_datas:
            prompt = prompt_text + '\n\nEvidence: {0}\nQuestion: {1}\nA. {2}\nB. {3}\nC. {4}\nD. {5}\nAnswer:'.format(data['temporal_conflict_evidence'], data['question'], data['options'][0], data['options'][1], data['options'][2], data['options'][3])
            data = {'prompt': prompt,
                    'true label': data['correct_option'],
                    'replaced label': data['replaced_option']}
            json_data = json.dumps(data)
            f.write(json_data+'\n')

    ### temporal_description
    out_file = out_dir+'/'+'temporal_description.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        for data in all_datas:
            if len(data['conflict_time_span']) == 1:
                prompt = prompt_text + '\n\nEvidence: {0}\nQuestion: {1} in {2}\nA. {3}\nB. {4}\nC. {5}\nD. {6}\nAnswer:'.format(data['temporal_conflict_evidence'], data['question'], data['conflict_time_span'][0], data['options'][0], data['options'][1], data['options'][2], data['options'][3])
            else:
                prompt = prompt_text + '\n\nEvidence: {0}\nQuestion: {1} from {2} to {3}\nA. {4}\nB. {5}\nC. {6}\nD. {7}\nAnswer:'.format(data['temporal_conflict_evidence'], data['question'], data['conflict_time_span'][0], data['conflict_time_span'][1], data['options'][0], data['options'][1], data['options'][2], data['options'][3])
            data = {'prompt': prompt,
                    'true label': data['correct_option'],
                    'replaced label': data['replaced_option']}
            json_data = json.dumps(data)
            f.write(json_data+'\n')

    ### semantic
    out_file = out_dir+'/'+'semantic.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        for data in all_datas:
            prompt = prompt_text + '\n\nEvidence: {0}\nQuestion: {1}\nA. {2}\nB. {3}\nC. {4}\nD. {5}\nAnswer:'.format(data['semantic_conflict_evidence'], data['question'], data['options'][0], data['options'][1], data['options'][2], data['options'][3])
            data = {'prompt': prompt,
                    'true label': data['correct_option'],
                    'replaced label': data['replaced_option']}
            json_data = json.dumps(data)
            f.write(json_data+'\n')

    ### semantic_description
    out_file = out_dir+'/'+'semantic_description.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        for data in all_datas:
            prompt = prompt_text + '\n\nEvidence: {0}\nQuestion: {1}\n{2} in this quesion means {3}\nA. {4}\nB. {5}\nC. {6}\nD. {7}\nAnswer:'.format(data['semantic_conflict_evidence'], data['question'], data['subject'], data['semantic_description'], data['options'][0], data['options'][1], data['options'][2], data['options'][3])
            data = {'prompt': prompt,
                    'true label': data['correct_option'],
                    'replaced label': data['replaced_option']}
            json_data = json.dumps(data)
            f.write(json_data+'\n')

    ### misinformation
    out_file = out_dir+'/'+'misinformation.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        for data in all_datas:
            prompt = prompt_text + '\n\nEvidence: {0}\nQuestion: {1}\nA. {2}\nB. {3}\nC. {4}\nD. {5}\nAnswer:'.format(data['fact_conflict_evidence'], data['question'], data['options'][0], data['options'][1], data['options'][2], data['options'][3])
            data = {'prompt': prompt,
                    'true label': data['correct_option'],
                    'replaced label': data['replaced_option']}
            json_data = json.dumps(data)
            f.write(json_data+'\n')

    ### correct + misinformation
    out_file = out_dir+'/'+'correct_misinformation.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        for data in all_datas:
            prompt = prompt_text + '\n\nEvidence: {0}\n{1}\nQuestion: {2}\nA. {3}\nB. {4}\nC. {5}\nD. {6}\nAnswer:'.format(data['correct_evidence'], data['fact_conflict_evidence'], data['question'], data['options'][0], data['options'][1], data['options'][2], data['options'][3])
            data = {'prompt': prompt,
                    'true label': data['correct_option'],
                    'replaced label': data['replaced_option']}
            json_data = json.dumps(data)
            f.write(json_data+'\n')

    ### correct + semantic_description
    out_file = out_dir+'/'+'correct_semantic.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        for data in all_datas:
            prompt = prompt_text + '\n\nEvidence: {0}\n{1}\nQuestion: {2}\nA. {3}\nB. {4}\nC. {5}\nD. {6}\nAnswer:'.format(data['correct_evidence'], data['semantic_conflict_evidence'], data['question'], data['options'][0], data['options'][1], data['options'][2], data['options'][3])
            data = {'prompt': prompt,
                    'true label': data['correct_option'],
                    'replaced label': data['replaced_option']}
            json_data = json.dumps(data)
            f.write(json_data+'\n')

    ### correct + semantic_description
    out_file = out_dir+'/'+'correct_semantic_description.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        for data in all_datas:
            prompt = prompt_text + '\n\nEvidence: {0}\n{1}\nQuestion: {2}\n{3} in this quesion means {4}\nA. {5}\nB. {6}\nC. {7}\nD. {8}\nAnswer:'.format(data['correct_evidence'], data['semantic_conflict_evidence'], data['question'], data['subject'], data['semantic_description'], data['options'][0], data['options'][1], data['options'][2], data['options'][3])
            data = {'prompt': prompt,
                    'true label': data['correct_option'],
                    'replaced label': data['replaced_option']}
            json_data = json.dumps(data)
            f.write(json_data+'\n')

    ### correct + temporal
    out_file = out_dir+'/'+'correct_temporal.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        for data in all_datas:
            prompt = prompt_text + '\n\nEvidence: {0}\n{1}\nQuestion: {2}\nA. {3}\nB. {4}\nC. {5}\nD. {6}\nAnswer:'.format(data['correct_evidence'], data['temporal_conflict_evidence'], data['question'], data['options'][0], data['options'][1], data['options'][2], data['options'][3])
            data = {'prompt': prompt,
                    'true label': data['correct_option'],
                    'replaced label': data['replaced_option']}
            json_data = json.dumps(data)
            f.write(json_data+'\n')

    ### correct + temporal_description
    out_file = out_dir+'/'+'correct_temporal_description.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        for data in all_datas:
            if len(data['conflict_time_span']) == 1:
                prompt = prompt_text + '\n\nEvidence: {0}\n{1}\nQuestion: {2} in {3}\nA. {4}\nB. {5}\nC. {6}\nD. {7}\nAnswer:'.format(data['correct_evidence'], data['temporal_conflict_evidence'], data['question'], data['conflict_time_span'][0], data['options'][0], data['options'][1], data['options'][2], data['options'][3])
            else:
                prompt = prompt_text + '\n\nEvidence: {0}\n{1}\nQuestion: {2} from {3} to {4}\nA. {5}\nB. {6}\nC. {7}\nD. {8}\nAnswer:'.format(data['correct_evidence'], data['temporal_conflict_evidence'], data['question'], data['conflict_time_span'][0], data['conflict_time_span'][1], data['options'][0], data['options'][1], data['options'][2], data['options'][3])
            data = {'prompt': prompt,
                    'true label': data['correct_option'],
                    'replaced label': data['replaced_option']}
            json_data = json.dumps(data)
            f.write(json_data+'\n')
            
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', help='')
    parser.add_argument('--out_dir', help='')
    args = parser.parse_args()
    main(args.in_file, args.out_dir)