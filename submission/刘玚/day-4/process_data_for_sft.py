#!/usr/bin/env python3
"""
将原始数据转换为 LLaMA-Factory 多轮对话格式，支持 function_call/observation。
将 function_call 合并到 gpt 回复中，确保满足 chat template 格式。
"""
import json
import os
from typing import List, Dict, Any

def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def process_data_for_multiturn(input_file: str, output_file: str):
    data = read_jsonl(input_file)
    processed_data = []
    for item in data:
        conversation = item.get('conversation', [])
        if len(conversation) < 2:
            continue
        messages = []
        # 1. user 问题
        user_msg = conversation[0].get('content', '')
        messages.append({"from": "human", "value": user_msg})
        
        # 2. assistant 第一次回复（合并 function_call）
        assistant_first = conversation[1].get('content', '')
        retrieve_args = item.get('retrieve_args', None)
        if retrieve_args is not None:
            # 将 function_call 合并到 gpt 回复中
            assistant_first += f"\n\n<function_calls>\n{retrieve_args}\n</function_calls>"
        messages.append({"from": "gpt", "value": assistant_first})
        
        # 3. observation（检索返回）
        retrieve_response = item.get('retrieve_response', None)
        if retrieve_response is not None:
            messages.append({"from": "observation", "value": retrieve_response})
        
        # 4. assistant 最终回复
        if len(conversation) > 3:
            assistant_final = conversation[3].get('content', '')
            messages.append({"from": "gpt", "value": assistant_final})
        
        processed_data.append({
            "conversations": messages,
            "system": "你是一个智能助手，能够判断是否需要使用搜索工具来回答问题，并能够正确处理搜索结果。"
        })
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    print(f"处理完成！共生成 {len(processed_data)} 条训练数据，保存到: {output_file}")

def create_dataset_info(output_dir: str):
    dataset_info = {
        "search_sft_data": {
            "file_name": "search_sft_data.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "system": "system"
            },
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
                "observation_tag": "observation",
                "function_tag": "function_call",
                "system_tag": "system"
            }
        }
    }
    dataset_info_path = os.path.join(output_dir, "dataset_info.json")
    with open(dataset_info_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    print(f"创建dataset_info.json: {dataset_info_path}")

def create_training_config(output_dir: str):
    config = {
        "model_name_or_path": "/remote-home1/share/models/Qwen/Qwen2.5-0.5B-Instruct/",
        "trust_remote_code": True,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "full",
        "deepspeed": "examples/deepspeed/ds_z3_config.json",
        "dataset": "search_sft_data",
        "dataset_dir": "/remote-home1/bwang/hw4/LLaMA-Factory/data",
        "template": "qwen",
        "cutoff_len": 2048,
        "max_samples": 1000,
        "overwrite_cache": True,
        "preprocessing_num_workers": 1,
        "dataloader_num_workers": 1,
        "output_dir": "saves/qwen2.5-0.5b/full/sft",
        "logging_steps": 10,
        "save_steps": 500,
        "plot_loss": True,
        "overwrite_output_dir": True,
        "save_only_model": False,
        "report_to": "none",
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1.0e-5,
        "num_train_epochs": 3.0,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "bf16": True,
        "ddp_timeout": 180000000,
        "resume_from_checkpoint": False
    }
    config_path = os.path.join(output_dir, "qwen_full_sft.yaml")
    with open(config_path, 'w', encoding='utf-8') as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    print(f"创建训练配置文件: {config_path}")

if __name__ == "__main__":
    input_file = "test_new_generator_data.jsonl"
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "search_sft_data.json")
    process_data_for_multiturn(input_file, output_file)
    create_dataset_info(output_dir)
    create_training_config(".")
    print("所有文件创建完成！") 