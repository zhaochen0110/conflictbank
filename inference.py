import json
import os
import argparse
import torch
import glob
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import numpy as np

def inference(model_name, input_dir, out_dir):
    """
    Run inference using the specified model on the input JSON files and save the results.

    :param model_name: str, name of the pre-trained model
    :param input_dir: str, directory containing input JSON files
    :param out_dir: str, directory to save the output JSON files
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=False, use_fast=False, trust_remote_code=True)
    print(f"Accessing {torch.cuda.device_count()} GPUs!")

    llm = LLM(model=model_name, dtype="float16", tensor_parallel_size=torch.cuda.device_count(), trust_remote_code=True)
    sampling_params = SamplingParams(logprobs=1000)
    
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    output_dir = os.path.join(out_dir, os.path.basename(model_name))
    os.makedirs(output_dir, exist_ok=True)
    out_files = glob.glob(os.path.join(output_dir, '*.json'))
    processed_files = [os.path.basename(file) for file in out_files]

    for input_file in tqdm(json_files, desc="Processing files"):
        if os.path.basename(input_file) in processed_files:
            continue
        
        prompts = []
        all_datas = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                prompts.append(data['prompt'])
                all_datas.append(data)
        
        for i in range(0, len(all_datas), 1000):
            batch_prompts = prompts[i:i + 1000]
            output = llm.generate(batch_prompts, sampling_params)
        
            predictions = []
            probs = []
            for num in range(len(output)):
                candidate_logits = []
                for label in [" A", " B", " C", " D"]:
                    try:
                        label_ids = tokenizer.encode(label, add_special_tokens=False)
                        label_id = label_ids[-1]
                        candidate_logits.append(output[num].outputs[0].logprobs[0][label_id].logprob)
                    except:
                        print(f"Warning: {label} not found. Artificially adding log prob of -100.")
                        candidate_logits.append(-100)
                
                candidate_logits = torch.tensor(candidate_logits).to(torch.float32)
                prob = torch.nn.functional.softmax(candidate_logits, dim=0).detach().cpu().numpy()
                answer = {i: k for i, k in enumerate(["A", "B", "C", "D"])}[np.argmax(prob)]
                predictions.append(answer)
                probs.append({'A': float(prob[0]), 'B': float(prob[1]), 'C': float(prob[2]), 'D': float(prob[3])})
            
            output_file = os.path.join(output_dir, os.path.basename(input_file))
            with open(output_file, 'a', encoding='utf-8') as f:
                for j in range(1000):
                    if i + j >= len(all_datas):
                        break
                    data = all_datas[i + j]
                    data['prediction'] = predictions[j]
                    data['prob'] = probs[j]
                    f.write(json.dumps(data) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference on input JSON files using a specified model.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the pre-trained model.")
    parser.add_argument('--input_dir', type=str, required=True, help="Directory containing input JSON files.")
    parser.add_argument('--out_dir', type=str, required=True, help="Directory to save the output JSON files.")
    args = parser.parse_args()

    inference(args.model_path, args.input_dir, args.out_dir)
