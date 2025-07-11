from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class FakeSearch:
    def __init__(self, model_name="/inspire/hdd/project/embodied-multimodality/public/syfei/baseline-models/Qwen3-8B"):
        # 加载本地模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def chat(self, messages: list):
        # 将消息格式化为模型输入
        input_text = "\n".join([msg["content"] for msg in messages])
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt')

        # 生成模型输出
        with torch.no_grad():
            output = self.model.generate(input_ids, max_length=12000, temperature=0.5, num_return_sequences=1)
        
        # 解码输出
        result = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return result

    def search(self, keyword, top_k=3):
        res = self.chat([{
            "role": "user",
            "content": f"请你扮演一个搜索引擎，对于任何的输入信息，给出 {min(top_k, 10)} 个合理的搜索结果，以列表的方式呈现。列表由空行分割，每行的内容是不超过500字的搜索结果。\n\n输入: {keyword}"
        }])
        
        # 处理模型返回的结果
        res_list = res.split("\n")  # 假设每个结果在新的一行
        return [res.strip() for res in res_list if len(res) > 0][:top_k]

if __name__ == "__main__":
    import sys
    search = FakeSearch()
    print(search.search(sys.argv[1]), 5)