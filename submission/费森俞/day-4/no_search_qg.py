import json
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# 加载模型
model_name = "/inspire/hdd/project/embodied-multimodality/public/syfei/baseline-models/DeepSeek-R1-Distill-Qwen-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 非检索类问题模板（固定知识/主观思考/常识类）
NO_SEARCH_PROMPT_TEMPLATES = [
    "请生成一个关于{category}的理论性问题，答案应基于公认原理而非实时数据。",
    "设计一个{category}领域的知识测试题，考察基础概念而非最新动态。",
    "提出一个不需要外部验证的{category}思考题，答案可通过逻辑推导得出。",
    "创建一个基于数学/逻辑的{category}问题，有明确的标准答案。",
    "构思一个关于{category}的经典问题，答案在教科书或百科中可找到。"
]

# 非检索问题的分类（固定知识领域）
NO_SEARCH_CATEGORIES = [
    "数学定理", "物理定律", "化学方程式", "编程语法",
    "历史事件", "文学名著", "哲学思想", "艺术理论",
    "地理常识", "语法规则", "逻辑谜题", "经济原理",
    "心理学基础", "音乐理论", "语言学习", "统计学方法",
    "经典算法", "会计原则", "教育理论", "烹饪原理"
]

def generate_no_search_question(category):
    """生成不需要检索的问题"""
    prompt_template = random.choice(NO_SEARCH_PROMPT_TEMPLATES)
    prompt = prompt_template.format(category=category)
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(
        input_ids,
        max_length=80,
        num_return_sequences=1,
        do_sample=True,
        top_p=0.85,  # 稍低的随机性保证问题严谨性
        temperature=0.5,
        no_repeat_ngram_size=4
    )
    question = tokenizer.decode(output[0], skip_special_tokens=True)
    return question.replace(prompt, "").strip()

def generate_non_search_questions(num_questions=500):
    questions = set()
    pbar = tqdm(total=num_questions, desc="生成非检索问题")
    
    while len(questions) < num_questions:
        category = random.choice(NO_SEARCH_CATEGORIES)

        try:
            question = generate_no_search_question(category)
            
            # 更严格的质量控制（非检索问题需更严谨）
            if (10 < len(question) <= 150) and any(mark in question for mark in ["？", "?", "吗", "呢"]) and not any(time_word in question for time_word in ["今年", "最近", "最新", "202"]):
                questions.add(question)
                pbar.update(1)
                
                if len(questions) % 50 == 0:
                    with open('no_search_questions_temp.json', 'w', encoding='utf-8') as f:
                        json.dump([{"question": q, "needs_search": False} for q in questions], 
                                f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"生成问题时出错: {e}")
            continue
    
    pbar.close()
    return [{"question": q, "needs_search": False} for q in questions]

# 生成并保存问题
questions = generate_non_search_questions(1)
with open('non_search_questions.json', 'w', encoding='utf-8') as f:
    json.dump(questions, f, ensure_ascii=False, indent=2)

print(f"已生成{len(questions)}个非检索型问题，保存至non_search_questions.json")