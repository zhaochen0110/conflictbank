
# **ConflictBank** 🏦
## 🎉 Overview

Welcome to **ConflictBank 🏦**, the first comprehensive benchmark for analyzing models' behavior by simulating knowledge conflicts encountered during pre-training and inference stages. This benchmark includes **7,453,853** claim-evidence pairs and **553,117** QA pairs, covering three main conflict causes: misinformation conflict, temporal conflict, and semantic conflict.

This repository includes scripts for downloading the ConflictBank datasets, as well as scripts for evaluating and training models on ConflictBank 🏦.

## ⚙️ **Installation**

Clone this repository and install the required packages:

```bash
git clone https://github.com/zhaochen0110/conflictbank.git
cd conflictbank
pip install -r requirements.txt
```

## **📊 Quick Start**

### **🚧 Data Loading**

You can obtain ConflictBank through the following code:

```python
from datasets import load_dataset
# load the claim-evidence pairs
dataset = load_dataset("Warrieryes/CB_claim_evidence")
# load the QA pairs
dataset = load_dataset("Warrieryes/CB_qa")
```

### **💎 Large-scale Evaluation**

To replicate the experimental results in our paper, run:

```bash
python inference.py --model_path "$model_path" \
--input_dir "$input_dir" \
--out_dir "$out_dir" 

python evaluate.py "$input_dir/$model_name" "$out_dir" 
```

### 🚅 Continual Pre-training

For further analysis, we have released all the LLMs with embedded conflicts used in our experimental analysis.

You can also train your own model using [llama_factory](https://github.com/hiyouga/LLaMA-Factory) with the following training script:

```bash
python src/train_bash.py \
--stage pt \
--model_name_or_path $MODEL_PATH \
--do_train \
--dataset $DATA_PATH \
--overwrite_cache \
--finetuning_type full \
--output_dir $OUTPUT_PATH \
--per_device_train_batch_size 32 \
--gradient_accumulation_steps 1 \
--lr_scheduler_type cosine \
--use_fast_tokenizer \
--logging_steps 1 \
--max_steps 4500 \
--learning_rate 2e-5 \
--fsdp "shard_grad_op auto_wrap" \
--fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
--save_steps 10000 \
--flash_attn \
--save_safetensors False \
--plot_loss \
--report_to wandb \
--overwrite_output_dir \
--cache_path $TOKENIZER_PATH \
--max_length 4096 \
--bf16
```

## **🏗️ Datasets Construction**

We also provide a detailed pipeline to generate our dataset from scratch.

### Step 1: Download and Setup Wikidata

First, you'll need to install SLING, a framework for natural language frame semantics, and fetch the relevant datasets:

```bash
# Install SLING via pip.
pip3 install https://ringgaard.com/data/dist/sling-3.0.0-py3-none-linux_x86_64.whl

# Download SLING KB and en wikidata mapping.
sling fetch --dataset kb,mapping --overwrite
```

### Step 2: Fact Extraction and Conflict Claim Construction

Next, use the provided script to extract facts and construct conflict claims:

```bash
python3 data_construct.py \
    --qid_names_file "$QID_NAMES_FILE" \
    --kb_file "$KB_FILE" \
    --fact_triples_file "$FACT_TRIPLES_FILE" \
    --templates_file "$TEMPLATES_FILE" \
    --conflict_row_output_file "$CONFLICT_ROW_OUTPUT_FILE" \
    --relation_to_object_output_file "$RELATION_TO_OBJECT_OUTPUT_FILE" \
    --s_r_object_output_file "$S_R_OBJECT_OUTPUT_FILE" \
    --fact_conflict_output_file "$FACT_CONFLICT_OUTPUT_FILE"

```

### Step 3: Generate Conflict Evidence

Generate evidence for the conflicts using a pre-trained model. Specify the type of conflict you want to simulate:

```bash
CONFLICT_TYPE="semantic_conflict" # Options: 'correct', 'fact_conflict', 'temporal_conflict', 'semantic_conflict'

python3 generate_conflicts.py \
    --model_name "$MODEL_NAME" \
    --file_path "$FILE_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_batch "$NUM_BATCH" \
    --conflict_type "$CONFLICT_TYPE"

```

### Step 4: Quality Control

Finally, run the quality control script to ensure the dataset's integrity and quality:

We use the [**deberta-v3-base-tasksource-nli**](https://huggingface.co/sileod/deberta-v3-base-tasksource-nli) model for NLI tasks and the [**all-mpnet-base-v2**](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) model for sentence embeddings.

You can get the trained confirmation model from this link.

```bash
python quality_control.py --raw_data_dir \
    --nli_model deberta-v3-base-tasksource-nli \
    --embedding_model all-mpnet-base-v2 \
    --classifier_model sbert_conflict_dict.pth \
    --selected_raw_data_path selected_data.json \
    --question_template_path question_templates.json \
    --output_path test_dataset.json \
    --relation_to_object relation_to_object.json\
    --qid_names qid_names.txt \
    --batch_size 32
```

## 📬 Contact

For any questions or inquiries, please feel free to open an issue or contact us at [suzhaochen0110@gmail.com].

## 🤝 Contributing

We welcome contributions to ConflictBank! If you have any suggestions or improvements, please open a pull request or contact us directly.

## 📜 License

This project is licensed under the CC BY-SA 4.0 license - see the LICENSE file for details.


## **📖 Citation**

Please cite our paper if you use our data, model or code. Please also kindly cite the original dataset papers. 

```
@article{su2024conflictbank,
  title={Conflictbank: A benchmark for evaluating the influence of knowledge conflicts in llm},
  author={Su, Zhaochen and Zhang, Jun and Qu, Xiaoye and Zhu, Tong and Li, Yanshu and Sun, Jiashuo and Li, Juntao and Zhang, Min and Cheng, Yu},
  journal={arXiv preprint arXiv:2408.12076},
  year={2024}
}
```


