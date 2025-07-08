# SummerQuest-2025 Day-1-ljx

这是一个用于自动化批改 "SummerQuest-2025" Day-1 作业的 Python 脚本。因为我没有会员，该脚本目前使用**智谱AI (Zhipu AI)** 的 `glm-4` 模型，效果不太好。


## 环境配置  
### 1. 确保已安装 zhipuai 库 (pip install zhipuai)  

### 2. 设置环境变量 ZHIPU_API_KEY(export ZHIPU_API_KEY='08ddb884374c4e28ac573d7afd6c91bb.XYRJk7eaTandlzUB')  

### 3. 在命令行中运行: python TA.py <path_to_submission_directory>(python TA.py ../../../submission)



## 目录结构

```
day-1/李佳羲
├── TA.py                 # 主执行脚本
├── prompts/
│   └── day-1.md          # 定义批改逻辑的Prompt模板
├── reports/              # 执行时生成的存放所有批改报告的目录
└── README.md             # 本说明文件
```

## 修改脚本

- **修改批改标准**: 直接编辑 `prompts/day-1.md` 文件。
- **跳过更多学生**: 在 `TA.py` 脚本顶部的 `SKIP_LIST` 列表中添加更多学生姓名。
- **更换大模型**: 修改 `TA.py` 顶部的 `MODEL_ID` 变量。
- **更改必需文件**: 修改 `TA.py` 顶部的 `REQUIRED_FILES` 列表，以适应不同的作业要求。
