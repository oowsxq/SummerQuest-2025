#!/usr/bin/env python3
"""
测试重构后的SearchDistillationDataGenerator功能
"""

from distill import SearchDistillationDataGenerator

def test_new_generator():
    """测试新的数据生成器"""
    generator = SearchDistillationDataGenerator("/remote-home1/share/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    
    # 测试单个查询
    query = "今天北京的天气怎么样？"
    
    print("="*60)
    print("测试查询:", query)
    print("="*60)
    
    # 第一步：判断是否需要检索
    print("\n1. 判断是否需要检索:")
    decision_prompt = generator.generate_search_decision_prompt(query)
    print("Prompt:", decision_prompt[:200] + "...")
    
    # 第二步：生成检索器调用
    print("\n2. 生成检索器调用:")
    retrieve_prompt = generator.generate_retrieve_call_prompt(query)
    print("Prompt:", retrieve_prompt[:200] + "...")
    
    # 第三步：模拟检索器
    print("\n3. 模拟检索器调用:")
    retrieve_response = generator.simulate_retriever(query)
    print("检索结果:", retrieve_response)
    
    # 第四步：生成最终回答
    print("\n4. 生成最终回答:")
    final_prompt = generator.generate_final_answer_prompt(query, retrieve_response)
    print("Prompt:", final_prompt[:200] + "...")
    
    # 生成完整训练数据
    print("\n" + "="*60)
    print("生成完整训练数据:")
    print("="*60)
    
    training_data = generator.generate_training_data(num_samples=2)
    
    for i, data in enumerate(training_data):
        print(f"\n样本 {i+1}:")
        print(f"查询: {data['query']}")
        print(f"需要检索: {data['needs_retrieve']}")
        print(f"检索参数: {data['retrieve_args']}")
        print(f"检索结果: {data['retrieve_response'][:100]}...")
        print(f"对话长度: {len(data['conversation'])} 条消息")
        
        # 保存到文件
        generator.save_training_data(training_data, "test_new_generator_data.jsonl")
        print(f"\n数据已保存到 test_new_generator_data.jsonl")

if __name__ == "__main__":
    test_new_generator() 