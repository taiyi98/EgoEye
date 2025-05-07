import cv2
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO

def plot_gaze_trajectory(image_path, gaze_info_list, output_image_path="gaze_trajectory.png", sigma=20, radius=60, weight_factor=20, alpha=0.5):
    """
    创建眼动轨迹的显著性图（Salience Map），调整影响区域和热度区分度
    :param image_files: 包含9帧图像路径的列表
    :param gaze_info_list: 每帧的gaze信息列表，格式为 [{'gaze_x': x, 'gaze_y': y}, ...]
    :param output_image_path: 输出的显著性图保存路径
    :param sigma: 高斯模糊的标准差，控制热图的平滑度
    :param radius: 影响区域的半径（控制每个眼动点的影响范围）
    :param weight_factor: 控制热度增量的因子，增加热度的区分度
    :return: 返回base64格式的显著性图
    """

    first_image = cv2.imread(image_path)
    height, width, _ = first_image.shape  

    
    salience_map = np.zeros((height, width), dtype=np.float32)

    
    step_values = np.linspace(10, 100, len(gaze_info_list))  # 生成从10到100的递增序列

    for i, gaze_info in enumerate(gaze_info_list):
        gaze_x = int(gaze_info['gaze_x'] * width)  # 转换为像素坐标
        gaze_y = int(gaze_info['gaze_y'] * height)

        # 为每个注视点创建一个影响区域，每隔step_size像素增加一次
        for dx in range(-radius, radius + 1, 10):  # 每隔step_size像素增加一次
            for dy in range(-radius, radius + 1, 10):  # 每隔step_size像素增加一次
                # 计算当前像素点的距离
                dist = np.sqrt(dx ** 2 + dy ** 2)
                if dist <= radius:
                    # 计算影响的热度，使用高斯衰减函数或距离衰减
                    target_x = gaze_x + dx
                    target_y = gaze_y + dy
                    
                    # 如果越界，取边界值
                    target_x = max(0, min(target_x, salience_map.shape[1] - 1))
                    target_y = max(0, min(target_y, salience_map.shape[0] - 1))
                    
                    # 更新热度图
                    salience_map[target_y, target_x] += weight_factor * np.exp(-dist ** 2 / (2 * (sigma ** 2)))
                
                # 增加热度的区分度，按权重增加热度
                gaze_x = max(0, min(gaze_x, salience_map.shape[1] - 1))
                gaze_y = max(0, min(gaze_y, salience_map.shape[0] - 1))

        # 增加热度的区分度，按递增的百分制权重增加热度
        salience_map[gaze_y, gaze_x] += weight_factor * step_values[i]  # 使用百分制阶梯递增值
    ksize = int(6 * sigma + 1)  # 确保ksize为奇数
    if ksize % 2 == 0:
        ksize += 1
    # 应用高斯平滑处理（控制显著性热图的平滑度）
    salience_map = cv2.GaussianBlur(salience_map, (ksize, ksize), 0)

    # 将热图归一化到[0, 255]范围
    salience_map_normalized = cv2.normalize(salience_map, None, 0, 255, cv2.NORM_MINMAX)

    # 将归一化后的热图转换为RGB格式
    # heatmap = cv2.applyColorMap(salience_map_normalized.astype(np.uint8), cv2.COLORMAP_JET)


    # 将热力图叠加到最后一帧图像上，使用alpha融合
    # overlay = cv2.addWeighted(first_image, 1 - alpha, heatmap, alpha, 0)
    # cv2.imwrite("salience_map.png", salience_map_normalized)

    # cv2.imwrite(f"Heat_line_sig{sigma}_rad{radius}_factor{weight_factor}.png", heatmap)

    # 直接保存图像（不显示标题和坐标）
    # cv2.imwrite(f"Heat_line_sig{sigma}_rad{radius}_factor{weight_factor}.png", overlay)

    # 可视化显著性图
    # plt.figure(figsize=(10, 6))
    # plt.imshow(salience_map, cmap='jet', interpolation='nearest')  # 使用'jet'色图显示热图


    # 保存图像
    # plt.savefig(output_image_path)
    # print(f"Salience Map of Gaze Trajectory saved to {output_image_path}")

    image_stream = BytesIO()
    _, encoded_img = cv2.imencode('.png', salience_map_normalized.astype(np.uint8))  # 直接用归一化后的图像编码
    image_stream.write(encoded_img)
    image_stream.seek(0)
    image_base64 = base64.b64encode(image_stream.read()).decode('utf-8')

    return image_base64

