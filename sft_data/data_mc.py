import pandas as pd
import json

# 读取你的原始CSV文件
input_file = '/home/pty_ssd/Qwen2-VL/QA_benchmark0226/egtea_temporal.csv'  # <<< 这里换成你的文件路径
output_file = 'temporal_egtea.json'  # 输出的JSON文件路径

# 读入数据
df = pd.read_csv(input_file)

# 存放最终结果的列表
output_data = []

for idx, row in df.iterrows():
    video_id = row['video_id']
    group_id = row['group_id']
    question = row['Question']
    answer_options = row['Answer Options']
    correct_answer = row['Correct Answer']

    # 处理 group_id -> 图片列表
    images = [f"egtea/{video_id}/{img.strip()}" for img in group_id.replace('\r', '').split('\n') if img.strip()]

    # 处理 Answer Options -> 按行拆
    options = answer_options.split('\n')
    options_text = "\n".join(options).strip()  # 保持格式，每个选项单独一行

    # 构建 user content: <image><image>... 问题文本 + 选项列表
    user_content = "".join(["<image>" for _ in images]) + question + "\n" + options_text

    # 注意：让 assistant 只输出 正确答案的字母，比如 "C"
    assistant_content = correct_answer.strip()

    # 构建 messages
    messages = [
        {
            "role": "user",
            "content": user_content
        },
        {
            "role": "assistant",
            "content": assistant_content
        }
    ]

    # 每个样本的字典
    sample = {
        "messages": messages,
        "images": images
    }

    output_data.append(sample)

# 写出到JSON文件
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print(f"✅ 转换完成！保存到了 {output_file}")
