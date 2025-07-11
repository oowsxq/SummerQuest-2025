import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# 加载模型
model_name = "/inspire/hdd/project/embodied-multimodality/public/syfei/baseline-models/DeepSeek-R1-Distill-Qwen-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 定义更具体的提示词模板
SEARCH_PROMPT_TEMPLATES = [
    "请生成一个需要检索最新信息的问题，关于{category}。问题应包含具体时间(如2023年)、地点或技术名称。",
    "创建一个需要验证事实的{category}问题，要求必须通过查阅权威资料才能回答。",
    "设计一个需要对比数据的{category}问题，要求比较两个实体在特定方面的差异。",
    "提出一个关于{category}的专业问题，需要查询数据库或学术论文才能准确回答。",
    "构思一个需要实时数据的{category}问题，答案会随时间变化。"
]

CATEGORIES = [
    "科技动态", "医学进展", "金融市场", "法律法规", 
    "体育赛事", "天气预报", "学术研究", "产品参数",
    "国际新闻", "经济指标", "交通信息", "健康指南",
    "教育政策", "就业数据", "旅游资讯", "文化事件",
    "环境监测", "食品安全", "能源价格", "军事动态"
]

def generate_search_question(category):
    """生成需要检索的问题"""
    import random
    prompt_template = random.choice(SEARCH_PROMPT_TEMPLATES)
    prompt = prompt_template.format(category=category)
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(
        input_ids,
        max_length=100,
        num_return_sequences=1,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        no_repeat_ngram_size=3
    )
    question = tokenizer.decode(output[0], skip_special_tokens=True)
    # 移除提示词部分，只保留生成的问题
    question = question.replace(prompt, "").strip()
    return question

def generate_questions(num_questions=500):
    questions = set()  # 使用集合自动去重
    pbar = tqdm(total=num_questions, desc="生成问题")
    
    while len(questions) < num_questions:
        # 随机选择一个类别
        category = random.choice(CATEGORIES)
        
        try:
            question = generate_search_question(category)
            
            # 简单验证问题质量
            if len(question) > 10 and "?" in question and question not in questions:
                questions.add(question)
                pbar.update(1)
                
                # 每生成50个问题保存一次
                if len(questions) % 50 == 0:
                    with open('questions_temp.json', 'w', encoding='utf-8') as f:
                        json.dump([{"question": q, "needs_search": True} for q in questions], 
                                 f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"生成问题时出错: {e}")
            continue
    
    pbar.close()
    return [{"question": q, "needs_search": True} for q in questions]

# 生成问题并保存
import random
questions = generate_questions(1)
with open('need_search_questions.json', 'w', encoding='utf-8') as f:
    json.dump(questions, f, ensure_ascii=False, indent=2)

print(f"已生成{len(questions)}个不重复的问题，并保存到need_search_questions.json")