import json
import pandas as pd
import argparse
from vllm import LLM, SamplingParams
import random
import os


num_batch = 5000

model_name = "/mnt/petrelfs/share_data/quxiaoye/models/Meta-Llama-3-70B-Instruct"
llm = LLM(model=model_name, dtype="float16", tensor_parallel_size=8)
sampling_params = SamplingParams(temperature=0, max_tokens=50)


def get_last_processed_index(output_dir):
    all_files = [f for f in os.listdir(output_dir) if f.endswith('.jsonl')]
    files = sorted(all_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
    if not files:
        return -1 
    last_file = files[-1]
    last_index_num = (int(last_file.split('_')[1].split('.')[0]) + 1) * num_batch
    return last_index_num


for file_num in range(1,7):
    print(f"Read and process file{file_num}")
    data_batch = []
    file_path = f'/mnt/petrelfs/suzhaochen/knowledge_conflict/generated_based/datasets/conflict_dataset_{file_num}.json'

    output_dir = f"/mnt/petrelfs/suzhaochen/knowledge_conflict/generated_based/description/result_{file_num}"
    os.makedirs(output_dir, exist_ok=True)
    last_index_num = get_last_processed_index(output_dir)

    system_template = '<|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>'

    def generate_and_write(llm, sampling_params, data_batch, output_dir, index):
        prompts = [item['description_prompt'] for item in data_batch]
        outputs = llm.generate(prompts, sampling_params)
        for i, output in enumerate(outputs):
            
            output = output.outputs[0].text
            # print(output)
            if "<|start_header_id|>assistant<|end_header_id|>" in output:
                try:
                    result = output.split("<|start_header_id|>assistant<|end_header_id|>\n\n")[1]
                except IndexError:
                    result = output.split("<|start_header_id|>assistant<|end_header_id|>\n\n")[0]            
            else:
                result = output
            if "<|eot_id|>" in result:
                output_final = result.split("<|eot_id|>")[0]
            else:
                output_final = result
            data_batch[i]["semantic_description"] = output_final
            del data_batch[i]["description_prompt"] 

        output_file_path = os.path.join(output_dir, f"output_{(index // num_batch)}.jsonl")
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            for data in data_batch:
                output_file.write(json.dumps(data) + "\n")    


    with open(file_path, 'r', encoding='utf-8') as jsonl_file:
        for index, line in enumerate(jsonl_file):
            if index < last_index_num:
                continue

            item = json.loads(line)
            description_prompt = '''Task: Resolve semantic conflicts in descriptions involving the same terms used for different roles, due to polysemy. Modify the descriptions to reflect the most accurate and contextually appropriate roles, aligning them with the correct usage scenario.

Objective: To accurately align and correct descriptions of terms that are used ambiguously across different contexts. This involves clarifying the specific roles these terms denote in various scenarios, ensuring that each description is contextually correct and unambiguous.

Example:
- Correct Claim: Franck Dupont holds the position of conseiller municipal de Zouafques.
- Conflicting Claim: Franck Dupont holds the position of Governor of Taraba State.
- Original Description for "Franck Dupont": French politician.
- Description for "Governor of Taraba State": Political position in Nigeria.
- Task: Modify the description to modify the usage of "Franck Dupont" by aligning it with a role appropriate for "Governor of Taraba State".
- Modified Description for "Franck Dupont": Nigerian politician.

Template for Generating Descriptions:
- Correct Claim: {correct_pair}
- Conflicting Claim: {conflict_pair}
- Original Description for "{subject}": {subject_description}
- Description for "{replaced_object}": {object_description}
- Task: Modify the description to modify the usage of "{subject}" by aligning it with a role appropriate for "{replaced_object}".
- Modified Description for "{subject}": [Only return the answer]'''

            description_prompt = description_prompt.format(
                correct_pair=item['correct_pair'],
                conflict_pair=item['fact_conflict_pair'],
                subject=item['subject'],
                subject_description = item['subject_additional_information'],
                replaced_object=item['replaced_object'],
                object_description=item['replaced_object_additional_information']
            )
            item['description_prompt'] = description_prompt
            
            data_batch.append(item)

            if (index + 1) % num_batch == 0:
                print(f"Batch processed. Writing to output at index: {index}")
                generate_and_write(llm, sampling_params, data_batch, output_dir, index)
                data_batch = []

        if data_batch:
            generate_and_write(llm, sampling_params, data_batch, output_dir, index)
        
        