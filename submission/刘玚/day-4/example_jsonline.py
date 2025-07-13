#!/usr/bin/env python3
"""
演示jsonline格式的使用
"""

from distill import SearchDistillationDataGenerator
import json

def demo_jsonline():
    """演示jsonline格式的保存和读取"""
    
    # 初始化生成器
    generator = SearchDistillationDataGenerator("/remote-home1/share/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    
    # 生成少量数据用于演示
    print("生成训练数据...")
    training_data = generator.generate_training_data(num_samples=3)
    
    # 保存为jsonline格式
    filename = "demo_data.jsonl"
    generator.save_training_data(training_data, filename)
    
    # 读取jsonline文件
    print("\n读取jsonline文件...")
    loaded_data = generator.load_training_data(filename)
    
    # 验证数据
    print(f"\n验证数据:")
    print(f"原始数据条数: {len(training_data)}")
    print(f"加载数据条数: {len(loaded_data)}")
    
    # 显示第一条数据
    if loaded_data:
        print(f"\n第一条数据示例:")
        print(f"查询: {loaded_data[0]['query']}")
        print(f"需要检索: {loaded_data[0]['needs_retrieve']}")
        print(f"对话消息数: {len(loaded_data[0]['conversation'])}")
        
        # 显示对话内容
        print(f"\n对话内容:")
        for i, msg in enumerate(loaded_data[0]['conversation']):
            role = msg['role']
            content = msg.get('content', 'None')
            if len(content) > 100:
                content = content[:100] + "..."
            print(f"{i+1}. {role}: {content}")
    
    # 手动读取jsonline文件示例
    print(f"\n手动读取jsonline文件示例:")
    with open(filename, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 2:  # 只显示前2行
                break
            data = json.loads(line.strip())
            print(f"第{i+1}行: {data['query']}")

if __name__ == "__main__":
    demo_jsonline() 