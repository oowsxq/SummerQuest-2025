#!/usr/bin/env python3
"""
基于DeepSeek R1模型的蒸馏训练脚本
生成能够判断是否需要网络搜索并处理搜索结果的数据
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import random
from typing import List, Dict, Any
import re
from vllm import LLM, SamplingParams
import argparse

class SearchDistillationDataGenerator:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = LLM(model_path, tensor_parallel_size=1)
        
    def generate_search_decision_prompt(self, query: str) -> str:
        """生成判断是否需要使用检索器的prompt"""
        content = f"""下面的用户问题可能需要使用检索器回答，请你在回答之前：
1. 考虑一下该问题是否需要使用检索器。通常，需要使用检索器的场景包括，需要一些具有时效的信息，或者需要获得一些不知道的信息
2. 如果需要使用检索器，则根据检索器的规定字段，生成检索器调用请求
3. 综合之前推理信息和检索器调用回复结果，生成最终回答

用户的提问如下：{query}"""
        return self.tokenizer.apply_chat_template([{"role":"user","content":content}], tokenize=False)

    def generate_retrieve_call_prompt(self, query: str) -> str:
        """生成检索器调用的prompt"""
        content = f"""请根据用户问题调用搜索工具。

用户问题：{query}

请调用 search_web 工具，参数格式如下：
{{
    "query": "搜索关键词"
}}

请直接给出正确的工具调用。"""
        return self.tokenizer.apply_chat_template([{"role":"user","content":content}], tokenize=False)

    def generate_final_answer_prompt(self, query: str, retrieve_response: str) -> str:
        """生成最终回答的prompt"""
        content = f"""请根据用户问题和检索器调用结果，生成最终回答
用户问题：{query}
检索器调用结果：{retrieve_response}
最终回答："""
        return self.tokenizer.apply_chat_template([{"role":"user","content":content}], tokenize=False)

    def simulate_retriever(self, query: str) -> str:
        """模拟检索器调用结果"""
        content = f"""请你模拟一个检索器行为，根据用户query返回检索结果
用户query：{query}"""
        prompt = self.tokenizer.apply_chat_template([{"role":"user","content":content}], tokenize=False)
        response = self.model.generate([prompt], sampling_params=SamplingParams(temperature=0.7, max_tokens=200))
        return response[0].outputs[0].text

    def generate_training_data(self, num_samples: int = 10) -> List[Dict[str, Any]]:
        """生成训练数据"""
        training_data = []
        
        # 需要搜索的查询模板
        search_queries = [
            "今天北京的天气怎么样？",
            "最新的iPhone 15价格是多少？",
            "2024年奥斯卡最佳影片是什么？",
            "特斯拉股票今天的收盘价是多少？",
            "最近有什么热门电影上映？",
            "明天上海到北京的航班有哪些？",
            "最新的新冠疫情数据如何？",
            "苹果公司最新的财报怎么样？",
            "最近的NBA比赛结果如何？",
            "最新的科技新闻有哪些？",
            "今天的人民币汇率是多少？",
            "最近的演唱会信息有哪些？",
            "最新的游戏发布信息？",
            "最近的股市行情如何？",
            "最新的手机评测信息？"
        ]
        
        # 生成需要搜索的数据
        for i in range(num_samples):
            query = random.choice(search_queries)
            
            # 第一步：判断是否需要使用检索器
            decision_prompt = self.generate_search_decision_prompt(query)
            decision_response = self.model.generate([decision_prompt], sampling_params=SamplingParams(temperature=0.7, max_tokens=200))
            decision_result = decision_response[0].outputs[0].text
            
            # 正则匹配判断是否需要检索
            if re.search(r"需要.*检索|使用.*检索|调用.*检索", decision_result, re.IGNORECASE):
                need_retrieve = True
                
                # 第二步：生成检索器调用
                retrieve_prompt = self.generate_retrieve_call_prompt(query)
                retrieve_response = self.model.generate([retrieve_prompt], sampling_params=SamplingParams(temperature=0.7, max_tokens=200))
                retrieve_result = retrieve_response[0].outputs[0].text
                
                # 提取检索器参数
                import json
                try:
                    # 尝试从结果中提取JSON参数
                    json_match = re.search(r'\{[^}]*"query"[^}]*\}', retrieve_result)
                    if json_match:
                        retrieve_args = json_match.group()
                    else:
                        # 如果没有找到JSON，构造一个简单的参数
                        retrieve_args = '{"query": "' + query + '"}'
                except:
                    retrieve_args = '{"query": "' + query + '"}'
                
                # 第三步：模拟检索器调用结果
                retrieve_response_content = self.simulate_retriever(query)
                
            else:
                need_retrieve = False
                retrieve_result = ""
                retrieve_args = '{"query": ""}'
                retrieve_response_content = "[空，无需调用检索器]"
            
            # 第四步：生成最终回答
            final_prompt = self.generate_final_answer_prompt(query, retrieve_response_content)
            final_response = self.model.generate([final_prompt], sampling_params=SamplingParams(temperature=0.7, max_tokens=300))
            final_result = final_response[0].outputs[0].text
            
            # 构造完整的对话数据
            conversation_data: List[Dict[str, Any]] = [
                {"role": "user", "content": query},
                {"role": "assistant", "content": decision_result}
            ]
            
            if need_retrieve:
                conversation_data.append({
                    "role": "assistant", 
                    "content": None,
                    "tool_calls": [{
                        "type": "function",
                        "function": {
                            "name": "search_web",
                            "arguments": retrieve_args
                        }
                    }]
                })
            
            conversation_data.append({"role": "assistant", "content": final_result})
            
            training_data.append({
                "query": query,
                "conversation": conversation_data,
                "needs_retrieve": need_retrieve,
                "retrieve_args": retrieve_args,
                "retrieve_response": retrieve_response_content
            })
            
        return training_data
    

    

    def save_training_data(self, data: List[Dict[str, Any]], filename: str):
        """保存训练数据为jsonline格式"""
        with open(filename, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"训练数据已保存到 {filename} (jsonline格式)")
    
    def load_training_data(self, filename: str) -> List[Dict[str, Any]]:
        """从jsonline文件加载训练数据"""
        data = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # 跳过空行
                    data.append(json.loads(line.strip()))
        print(f"从 {filename} 加载了 {len(data)} 条训练数据")
        return data

def main():
    # 初始化数据生成器
    generator = SearchDistillationDataGenerator("/remote-home1/share/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    
    # 生成训练数据
    print("正在生成训练数据...")
    training_data = generator.generate_training_data(num_samples=10)
    import pdb;pdb.set_trace()
    
    # 保存数据
    generator.save_training_data(training_data, "search_distillation_data.jsonl")
    
    # 统计信息
    needs_retrieve_count = len([d for d in training_data if d["needs_retrieve"]])
    
    print(f"\n数据生成完成！")
    print(f"总样本数: {len(training_data)}")
    print(f"需要检索的样本: {needs_retrieve_count}")
    print(f"不需要检索的样本: {len(training_data) - needs_retrieve_count}")

def get_deepseek_search_data_dbg():
    llm = LLM("/remote-home1/share/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", tensor_parallel_size=1)
    tokenizer = AutoTokenizer.from_pretrained("/remote-home1/share/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "搜索网络获取最新信息，包括天气、新闻、价格等实时数据",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "搜索查询关键词"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]
    #判断是否需要使用检索器的prompt
    content = f"""下面的用户问题可能需要使用检索器回答，请你在回答之前：
    1. 考虑一下该问题是否需要使用检索器。通常，需要使用检索器的场景包括，需要一些具有时效的信息，或者需要获得一些不知道的信息
    2. 如果需要使用检索器，则根据检索器的规定字段，生成检索器调用请求
    3. 综合之前推理信息和检索器调用回复结果，生成最终回答
    
    用户的提问如下：今天北京的天气怎么样？"""
    #生成检索器调用的prompt
    get_retrieve_prompt = f"""
请根据用户问题调用搜索工具。

用户问题：今天北京的天气怎么样？

请调用 search_web 工具，参数格式如下：
{{
    "query": "搜索关键词"
}}

请直接给出正确的工具调用。
"""
    #生成最终回答的prompt
    get_answer_prompt = f"""
    请根据用户问题和检索器调用结果，生成最终回答
    用户问题：今天北京的天气怎么样？
    检索器调用结果：北京今天多云转晴，气温15-25°C，空气质量良好，适合户外活动。根据气象部门预报，北京今日天气稳定，无降水，风力较小。实时天气数据显示，北京当前温度20°C，湿度45%，能见度良好。
    最终回答：
    """

    messages = [{'role':'user',"content": get_retrieve_prompt}, {'role':'assistant',"content": None}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools)
    response = llm.generate([prompt], sampling_params=SamplingParams(temperature=1, max_tokens=1024))
    response_content = tokenizer.decode(response[0].outputs[0].token_ids, skip_special_tokens=False)
    # response_content = response[0].outputs[0].text
    print(response_content)
    print("--------------------------------")
    exit()

    messages.append({'role':'assistant',"content": response_content})
    messages.append({'role': 'assistant', 'content': None})
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools)
    print(prompt)
    print("--------------------------------")
    response = llm.generate([prompt], sampling_params=SamplingParams(temperature=1, max_tokens=1024))
    response_content = tokenizer.decode(response[0].outputs[0].token_ids, skip_special_tokens=False)
    print(response_content)
    
    # 不跳过特殊token进行解码
def render_chat_template():
    tokenizer = AutoTokenizer.from_pretrained("/remote-home1/share/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    
    # 定义工具
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "搜索网络获取最新信息，包括天气、新闻、价格等实时数据",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "搜索查询关键词"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]
    
    # 完整的对话消息
    messages = [
        {
            'role': 'system', 
            'content': '你是一个智能助手，能够分析用户查询，判断是否需要搜索，并基于搜索结果给出详细回答。'
        },
        {
            'role': 'user', 
            'content': '今天北京的天气怎么样？'
        },
        {
            'role': 'assistant', 
            'content': '让我分析一下这个查询。用户询问的是今天的天气情况，这需要获取实时信息，因为天气每天都在变化。我需要搜索获取最新的天气数据。'
        },
        {
            'role': 'assistant',
            'content': None,
            'tool_calls': [
                {
                    'type': 'function',
                    'function': {
                        'name': 'search_web',
                        'arguments': '{"query": "北京今天天气 实时"}'
                    }
                }
            ]
        },
        {
            'role': 'tool',
            'content': '北京今天多云转晴，气温15-25°C，空气质量良好，适合户外活动。根据气象部门预报，北京今日天气稳定，无降水，风力较小。实时天气数据显示，北京当前温度20°C，湿度45%，能见度良好。'
        },
        {
            'role': 'assistant',
            'content': '''基于搜索结果分析：

1. 信息可靠性：搜索结果来自多个权威来源，包括气象部门和实时数据，信息可靠。

2. 综合分析：
- 北京今天天气多云转晴，气温适宜（15-25°C）
- 空气质量良好，适合户外活动
- 无降水，风力较小，出行条件良好

3. 建议：
- 可以安排户外活动，但建议携带薄外套
- 注意防晒，紫外线指数中等
- 空气质量良好，适合运动

4. 补充说明：
如果计划具体活动，建议查看更详细的时段预报，因为天气可能在不同时段有所变化。'''
        }
    ]
    
    # 应用 chat template，传入 tools 参数
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        tools=tools  # 这里传入 tools
    )
    print(prompt)
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dbg", action="store_true", help="debug mode")
    parser.add_argument("--render", action="store_true", help="render chat template")
    args = parser.parse_args()
    if args.dbg:
        get_deepseek_search_data_dbg()
    elif args.render:
        render_chat_template()
    else:
        main()
