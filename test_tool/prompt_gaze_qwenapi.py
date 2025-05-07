import os
import json
import csv
import base64
import pandas as pd
from openai import OpenAI
from gaze_trajectory import plot_gaze_trajectory


def encode_images_from_folder(base_folder, video_id, group_id):
    """
    从指定 video_id 目录中获取 group_id 列表对应的图片，并编码为 Base64。
    :param base_folder: 母文件夹路径
    :param video_id: 视频 ID
    :param group_id: 图片文件名列表
    :return: Base64 编码后的图片列表
    """
    image_data_list = []
    folder_path = os.path.join(base_folder, video_id)
    
    for image_file in group_id:
        image_path = os.path.join(folder_path, image_file.strip())
        if os.path.exists(image_path):
            with open(image_path, "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode("utf-8")
                image_data_list.append(f"data:image/jpeg;base64,{base64_image}")
        else:
            print(f"警告: 找不到文件 {image_path}")
    
    return image_data_list, image_path


def get_gaze_info_from_csv(csv_folder, video_id, group_id):
    """
    :param csv_folder: 存放csv的文件夹路径
    :param video_id: 视频 ID
    :param group_id: 帧编号列表（如 ["138.jpg", ...]）
    :return: gaze 信息列表，包含每帧的 gaze_x 和 gaze_y
    """
    csv_file = os.path.join(csv_folder, f"{video_id}.csv")
    if not os.path.exists(csv_file):
        return None

    # 读取csv内容到字典
    gaze_dict = {}
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = row['frame']
            gaze_str = row['gaze'].strip('()')
            gaze_x, gaze_y = map(float, gaze_str.split(','))
            gaze_dict[frame] = {'gaze_x': gaze_x, 'gaze_y': gaze_y}

    # 按 group_id 顺序返回 gaze 信息
    gaze_info_list = []
    for image_file in group_id:
        gaze_info = gaze_dict.get(image_file)
        if gaze_info:
            gaze_info_list.append(gaze_info)
        else:
            gaze_info_list.append(None)  # 没有找到时返回 None

    return gaze_info_list


def main():
    datasets = ['egtea']
    categories = ['spatial', 'temporal', 'causal']
    for dataset in datasets:
        for category in categories:
            print(f"----Processing {category}_{dataset}.csv----\n")
            # 配置路径和 API 相关信息
            file_path = f"/home/pty_ssd/EgoEye/qa_pairs/{category}_{dataset}.csv"
            file_name = os.path.basename(file_path)
            new_file = os.path.splitext(file_name)[0]
            base_folder = f"/home/pty_ssd/EgoEye/datasets/{dataset}"
            gazees_folder = f"/home/pty_ssd/EgoEye/ablation/gazees_vllm/{dataset}"  # 替换为你的 JSON 文件路径
            api_key = "sk-fab6884dd83b4bafbec681b0ed885582"  # 替换为你的 API Key
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 替换为你的 Base URL
            
            df = pd.read_csv(file_path)
            client = OpenAI(api_key=api_key, base_url=base_url)
            
            results = []

            # 遍历 CSV 文件每一行，处理 VQA 任务
            for index, row in df.iterrows():
                video_id = row["video_id"]
                group_id = row["group_id"].split("\n")  # 解析图像文件名
                question = row["Question"]
                answer_options = row["Answer Options"]
                reference = row["Correct Answer"]

                # 编码图像
                image_data_list, image_path = encode_images_from_folder(base_folder, video_id, group_id)

                # 如果图片为空，跳过该条数据
                if not image_data_list:
                    print(f"跳过 {video_id}，未找到对应的图像文件")
                    continue

                # 获取 gaze_info 列表
                gaze_info_list = get_gaze_info_from_csv(gazees_folder, video_id, group_id)

                salience_image_base64 = plot_gaze_trajectory(image_path, gaze_info_list)
             

                input_question = ("I provide you with a Picture{Frame 0} and a video{Frame 1-9}. Choose the correct option based on the first-person perspective scene question.\n"
                                "You must follow these steps to answer the question:\n"
                                "1. {Frame 0} is the saliency grayscale map of the gaze trajectory from the first-person perspective, with gaze sequence from low brightness to high brightness.\n"
                                "2. Remember the location and time sequence of gaze areas in {Frame 0}.\n"
                                "3. Observe the video according to the gaze sequence in step 2.\n"
                                f"4. Question:\n{question}\nOptions:\n{answer_options}\n"
                                "Choose the most appropriate option. Only return the letter of the correct option.")


                
                try:
                    completion = client.chat.completions.create(
                        model="qwen-vl-max-latest",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image_url", "image_url":{"url":f"data:image/png;base64,{salience_image_base64}"}},
                                    {"type": "video", "video": image_data_list},
                                    {"type": "text", "text": input_question}
                                ]
                            }
                        ]
                    )
                    model_answer = completion.choices[0].message.content

                except Exception as e:
                    model_answer = f"API 调用失败: {e}"
                
                # print(f'{input_question}\nModel Answer: {model_answer} \nCorrect Answer: {reference}')

                results.append({
                    "video_id": video_id,
                    "question": question,
                    "answer_options": answer_options,
                    "model_answer": model_answer,
                    "reference_answer": reference
                })
                

            # 将结果转换为 DataFrame 并保存
            results_df = pd.DataFrame(results)
            results_df.to_csv(f"/home/pty_ssd/EgoEye/results/prompt_gazees/{new_file}.csv", index=False, encoding="utf-8")
            print(f"API 请求已完成，结果已保存至 {new_file}.csv。")



if __name__ == "__main__":
    main()
