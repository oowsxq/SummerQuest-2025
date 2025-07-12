import json
import torch
import random
from tqdm import tqdm
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM

from search_function import FakeSearch

class EnhancedAnswerGenerator:
    def __init__(self):
        # 初始化推理模型
        self.model_path = "/inspire/hdd/project/embodied-multimodality/public/syfei/baseline-models/DeepSeek-R1-Distill-Qwen-7B"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
        
        # 初始化伪搜索工具
        self.search_tool = FakeSearch()
        
        # 系统提示模板
        self.system_prompt = """你是一个具有网络搜索能力的AI助手。请按照以下步骤处理问题：
1. 分析问题类型
2. 决定是否需要搜索
3. 如需搜索，生成精确的搜索词
4. 综合信息给出最终答案

请用以下JSON格式响应：
{
    "thoughts": "你的思考过程",
    "action": {
        "needs_search": true/false,
        "search_query": "搜索词(如需要)"
    },
    "answer": "最终答案"
}"""
    
    def generate_full_response(self, question: str, ground_truth_needs_search: bool) -> Dict:
        """生成包含完整思考过程的响应"""
        # 第一步：让模型自主决策
        decision_prompt = f"""系统指令：{self.system_prompt}
        
用户问题：{question}
当前日期：2024年1月（模拟）"""
        
        # 生成初始思考过程
        initial_response = self._generate_response(decision_prompt)
        try:
            decision = json.loads(initial_response)
        except:
            decision = {"thoughts": "格式解析失败", "action": {"needs_search": False}, "answer": "无法处理该问题"}
        
        # 强制遵守预设的搜索需求（根据问题来源）
        decision["action"]["needs_search"] = ground_truth_needs_search
        decision["thoughts"] += f"\n（注：根据问题预设强制设置为{'需要' if ground_truth_needs_search else '不需要'}搜索）"
        
        # 第二步：执行搜索或直接回答
        if ground_truth_needs_search and decision["action"].get("search_query"):
            # 执行搜索
            search_results = self.search_tool.search(decision["action"]["search_query"])
            decision["action"]["search_results"] = search_results
            
            # 基于搜索生成最终答案
            answer_prompt = f"""系统指令：请根据以下搜索结果为问题提供专业回答：
问题：{question}
搜索结果：
{json.dumps(search_results, ensure_ascii=False)}

要求：
1. 标注信息来源
2. 如信息不足请明确说明"""
            final_answer = self._generate_response(answer_prompt)
            decision["answer"] = final_answer
        else:
            # 直接回答的二次验证
            if ground_truth_needs_search:
                decision["answer"] = "错误：此问题需要搜索但未执行搜索"
            else:
                verify_prompt = f"""验证以下回答是否包含时效信息（如需要请修改）：
问题：{question}
原回答：{decision['answer']}

请直接输出修正后的回答（如无需修改请重复原回答）"""
                decision["answer"] = self._generate_response(verify_prompt)
        
        return {
            "question": question,
            "metadata": {
                "preset_requires_search": ground_truth_needs_search,
                "model_decision": initial_response
            },
            "thought_process": decision["thoughts"],
            "actions": decision["action"],
            "final_answer": decision["answer"]
        }
    
    def _generate_response(self, prompt: str) -> str:
        """生成模型响应（带格式控制）"""
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.model.device)
        output = self.model.generate(
            input_ids,
            max_new_tokens=1024,
            temperature=0.3,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True).replace(prompt, "").strip()

def generate_training_dataset():
    """生成完整训练数据集"""
    generator = EnhancedAnswerGenerator()
    
    # 加载两类问题
    with open('non_search_questions.json', 'r') as f:
        non_search_questions = [{"question": q["question"], "needs_search": False} 
                              for q in json.load(f)]
    
    with open('need_search_questions.json', 'r') as f:
        search_questions = [{"question": q["question"], "needs_search": True} 
                          for q in json.load(f)]
    
    # 合并并打乱问题
    all_questions = non_search_questions + search_questions
    random.shuffle(all_questions)
    
    # 生成响应
    dataset = []
    for q in tqdm(all_questions, desc="生成训练数据"):
        try:
            response = generator.generate_full_response(q["question"], q["needs_search"])
            dataset.append(response)
            
            # 每10条保存一次
            if len(dataset) % 10 == 0:
                with open('enhanced_training_data.json', 'w', encoding='utf-8') as f:
                    json.dump(dataset, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"处理问题失败: {q['question']} - {str(e)}")
    
    # 最终保存
    with open('enhanced_training_data.json', 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    generate_training_dataset()