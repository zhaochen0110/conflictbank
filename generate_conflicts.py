import json
import argparse
import random
import os
from vllm import LLM, SamplingParams
from tqdm import tqdm

def generate_and_write(llm, sampling_params, data_batch, output_dir, index, num_batch, conflict_type):
    """
    Generate evidence using the language model and write the results to a file.
    
    :param llm: LLM, the language model
    :param sampling_params: SamplingParams, parameters for sampling
    :param data_batch: list, batch of data to process
    :param output_dir: str, directory to save the output files
    :param index: int, current index in processing
    :param num_batch: int, number of data samples per batch
    :param conflict_type: str, type of conflict (e.g., 'correct', 'fact_conflict', 'temporal_conflict', 'semantic_conflict')
    """
    prompts = [item[conflict_type + "_prompt"] for item in data_batch]
    outputs = llm.generate(prompts, sampling_params)
    for i, output in enumerate(outputs):
        output_text = output.outputs[0].text
        if "assistant" in output_text:
            try:
                result = output_text.split("assistant\n\n")[1]
            except IndexError:
                result = output_text.split("assistant\n\n")[0]            
        else:
            result = output_text
        output_final = result.split("")[0] if "" in result else result
        data_batch[i][conflict_type + "_evidence"] = output_final
        del data_batch[i][conflict_type + "_prompt"] 

    output_file_path = os.path.join(output_dir, f"output_{conflict_type}_{(index // num_batch)}.jsonl")
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for data in data_batch:
            output_file.write(json.dumps(data) + "\n")

def main(model_name, file_path, output_dir, num_batch, conflict_type):
    """
    Main function to process the input file and generate evidence using the language model.
    
    :param model_name: str, path to the pre-trained model
    :param file_path: str, path to the input JSON file
    :param output_dir: str, directory to save the output files
    :param num_batch: int, number of data samples per batch
    :param conflict_type: str, type of conflict (e.g., 'correct', 'fact_conflict', 'temporal_conflict', 'semantic_conflict')
    """
    # Initialize the language model with specific configurations
    llm = LLM(model=model_name, dtype="float16", tensor_parallel_size=8)
    sampling_params = SamplingParams(temperature=0, max_tokens=512)

    # Templates for generating evidence
    templates = {
        'correct': {
            "new": "Based on the provided description, craft a news report that offers well-researched and substantiated evidence for the statement. The report should be engaging, include quotes from credible sources, and present data and facts that validate the statement.",
            "book": "Utilizing the provided description, write a book narrative that intricately weaves in detailed evidence supporting the statement. The narrative should be rich in context, offer deep insights, and use storytelling to elucidate the facts that back the statement.",
            "wikipedia": "Following the provided description, create a Wikipedia entry that systematically presents comprehensive evidence for the statement. The entry should be informative, well-structured, and include references to credible sources that substantiate the statement."
        },
        'fact_conflict': {
            "new": "Based on the provided description, compose a news article that introduces a narrative aligning with the given claim, incorporating fictional interviews, events, and data. Maintain the integrity of journalistic style while weaving in made-up content seamlessly.",
            "book": "Using the provided description as a foundation, craft a section of a book narrative that subtly introduces elements that support the given claim. Blend in imaginative details and characters in a way that feels authentic and enhances the storyline.",
            "wikipedia": "Based on the provided description, construct a Wikipedia entry that outlines a series of events, studies, and references that are fictional but support the given claim. Ensure the entry maintains the formal tone and structure of a real Wikipedia article."
        },
        'temporal_conflict': {
            "new": "Based on the provided descriptions, please write a news report. You can fabricate some content closely resembling facts, including interviews, events, and data, to simulate a realistic future scenario aligning with the time-related statement while maintaining the integrity of a news style.",
            "book": "Using the provided description, write a narrative for a book, with a focus on the temporal information in the statement. Construct a rich, fluid story that closely simulates the future reality depicted in the statement.",
            "wikipedia": "Based on the provided description, construct a Wikipedia entry. Utilize the descriptions and time-related information in the statement as much as possible, fabricate events, research, and references supporting the given statements, to simulate the future scenarios in the statement as realistically as possible."
        },
        'semantic_conflict': {
            "new": "Based on the provided description, compose a news article that introduces a narrative aligning with the given claim, incorporating fictional interviews, events, and data. Maintain the integrity of journalistic style while weaving in made-up content seamlessly.",
            "book": "Using the provided description as a foundation, craft a section of a book narrative that subtly introduces elements that support the given claim. Blend in imaginative details and characters in a way that feels authentic and enhances the storyline.",
            "wikipedia": "Based on the provided description, construct a Wikipedia entry that outlines a series of events, studies, and references that are fictional but support the given claim. Ensure the entry maintains the formal tone and structure of a real Wikipedia article."
        }
    }

    os.makedirs(output_dir, exist_ok=True)

    data_batch = []

    with open(file_path, 'r', encoding='utf-8') as jsonl_file:
        for index, line in enumerate(jsonl_file):
            item = json.loads(line)
            key = random.choice(list(templates[conflict_type].keys()))
            template = templates[conflict_type][key]

            if conflict_type in ['correct', "fact_conflict", 'temporal_conflict'] and item.get("subject_additional_information"):
                template += f'Description for "{item["subject"]}": {item["subject_additional_information"]} '

            if conflict_type == 'semantic_conflict' and item.get('semantic_description'):
                template += f'Description for "{item["subject"]}": {item["semantic_description"]} '            

            object_ = item["replaced_object"] if conflict_type in ["fact_conflict", 'temporal_conflict', 'semantic_conflict'] else item["object"]

            if conflict_type == "correct" and item.get("object_additional_information"):
                template += f'Description for "{object_}": {item["object_additional_information"]} '
            if conflict_type in ["fact_conflict", 'temporal_conflict', 'semantic_conflict'] and item.get("replaced_object_additional_information"):
                template += f'Description for "{object_}": {item["replaced_object_additional_information"]} '

            claim_prompt = f"""Claim:
{item.get("fact_conflict_pair")}
Evidence:"""
            
            template += claim_prompt
            system_template = 'system\n\n{system_message}'
            template = system_template.format(system_message=template)
            item[f"{conflict_type}_prompt"] = template
            item[f"{conflict_type}_prompt_category"] = key

            data_batch.append(item)

            if (index + 1) % num_batch == 0:
                print(f"Batch processed. Writing to output at index: {index}")
                generate_and_write(llm, sampling_params, data_batch, output_dir, index, num_batch, conflict_type)
                data_batch = []

        if data_batch:
            generate_and_write(llm, sampling_params, data_batch, output_dir, index, num_batch, conflict_type)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate evidence using a pre-trained language model.")
    parser.add_argument('--model_name', type=str, required=True, help="Path to the pre-trained model.")
    parser.add_argument('--file_path', type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the output files.")
    parser.add_argument('--num_batch', type=int, default=5000, help="Number of data samples per batch.")
    parser.add_argument('--conflict_type', type=str, required=True, choices=['correct', 'fact_conflict', 'temporal_conflict', 'semantic_conflict'], help="Type of conflict to process.")
    args = parser.parse_args()

    main(args.model_name, args.file_path, args.output_dir, args.num_batch, args.conflict_type)
