# %%
import os
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager # 确保字体管理器已导入
from tqdm.autonotebook import tqdm # 用于显示进度


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 配置路径 ---
val_image_dir=      r'f:\val_img'
output_mask_dir=    r'result\result-4-15\Segmentation_Results'
output_mask_origin_output_dir = r'result\model_output' # 新增原始输出掩码路径
output_dir = r'output' # 新增保存输出图像的目录

# 创建输出目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 1. 原始验证集图像路径
#    (使用之前代码块中定义的 val_image_dir)
if 'val_image_dir' not in locals() or not os.path.isdir(val_image_dir):
    print("错误：原始验证集图像路径 'val_image_dir' 未定义或无效。请确保它已在之前的单元格中设置。")
    # 你可能需要根据实际情况手动设置:
    # val_image_dir = "/kaggle/input/pterygium/val_img/val_img" # Kaggle 示例
    # val_image_dir = "f:/val" # 本地示例
else:
    print(f"使用原始验证图像路径: {val_image_dir}")

# 2. 预测掩码图像路径 (包含模型生成的 RGB 掩码)
#    (使用之前代码块中定义的 output_mask_dir)
if 'output_mask_dir' not in locals() or not os.path.isdir(output_mask_dir):
    print("错误：预测掩码路径 'output_mask_dir' 未定义或无效。请确保它已在之前的单元格中设置，并且包含了预测生成的掩码文件。")
    # 你可能需要根据实际情况手动设置:
    # output_mask_dir = "/kaggle/working/mask" # Kaggle 示例
    # output_mask_dir = "/content/mask" # Colab 示例
    # output_mask_dir = "f:/mask" # 本地示例
else:
    print(f"使用预测掩码路径: {output_mask_dir}")

# 3. 原始输出掩码路径
if 'output_mask_origin_output_dir' not in locals() or not os.path.isdir(output_mask_origin_output_dir):
    print("错误：原始输出掩码路径 'output_mask_origin_output_dir' 未定义或无效。")
else:
    print(f"使用原始输出掩码路径: {output_mask_origin_output_dir}")

# --- 可视化函数 ---
def visualize_overlay(original_img_path, mask_img_path, origin_output_mask_path=None, save_path=None):
    """加载原图和掩码，生成灰度覆盖图并显示"""
    try:
        # 加载原图
        original_image = Image.open(original_img_path).convert("RGB")
        # 加载预测掩码 (RGB)
        mask_image = Image.open(mask_img_path).convert("RGB")

        # 检查尺寸是否一致 (可选，但推荐)
        if original_image.size != mask_image.size:
            print(f"警告：图像 {os.path.basename(original_img_path)} ({original_image.size}) "
                f"和掩码 {os.path.basename(mask_img_path)} ({mask_image.size}) 尺寸不匹配，跳过可视化。")
            return

        # 1. 将原图转为灰度，再转回 RGB 以便合并
        grayscale_image = original_image.convert("L")
        grayscale_rgb_image = grayscale_image.convert("RGB")
        grayscale_rgb_array = np.array(grayscale_rgb_image)

        # 2. 获取掩码数组
        mask_array = np.array(mask_image)

        # 3. 创建覆盖图像数组 (初始化为灰度RGB)
        overlay_array = grayscale_rgb_array.copy()

        # 4. 找到掩码中红色区域 (R=128, G=0, B=0) 的条件
        #    为增加鲁棒性，可以只检查 R 通道是否 > 0 (或接近 128)
        mask_condition = mask_array[:, :, 0] > 64 # 条件放宽一点，>64 应该能捕捉到 128

        # 5. 将掩码区域覆盖到灰度图上
        red_alpha = 0.4  # 不透明度
        overlay_array[mask_condition] = (red_alpha * mask_array[mask_condition] + 
                               (1 - red_alpha) * grayscale_rgb_array[mask_condition]).astype(np.uint8)
        
        # 处理第二个掩码（如果提供）
        if origin_output_mask_path and os.path.exists(origin_output_mask_path):
            # 加载原始输出掩码
            origin_output_mask = Image.open(origin_output_mask_path).convert("RGB")
            
            # 检查尺寸是否一致
            if original_image.size != origin_output_mask.size:
                print(f"警告：图像 {os.path.basename(original_img_path)} ({original_image.size}) "
                    f"和原始掩码 {os.path.basename(origin_output_mask_path)} ({origin_output_mask.size}) "
                    f"尺寸不匹配，不添加原始掩码。")
            else:
                # 获取原始掩码数组
                origin_mask_array = np.array(origin_output_mask)
                
                # 找到原始掩码中红色区域
                origin_mask_condition = origin_mask_array[:, :, 0] > 64
                
                # 创建绿色掩码 (0,255,128)
                green_mask = np.zeros_like(origin_mask_array)
                green_mask[origin_mask_condition, 1] = 255  # G=255
                green_mask[origin_mask_condition, 2] = 128  # B=128
                
                # 将绿色掩码区域覆盖到已有的覆盖图上
                green_alpha = 0.2  # 不透明度
                overlay_array[origin_mask_condition] = (green_alpha * green_mask[origin_mask_condition] + 
                                      (1 - green_alpha) * overlay_array[origin_mask_condition]).astype(np.uint8)

        # 转回 PIL Image
        overlay_image = Image.fromarray(overlay_array)
        
        # 保存结果图像（如果提供了保存路径）
        if save_path:
            overlay_image.save(save_path)

    except FileNotFoundError:
        print(f"错误：文件未找到 - {original_img_path} 或 {mask_img_path}")
    except Exception as e:
        print(f"处理图像 {os.path.basename(original_img_path)} 时发生错误: {e}")


# --- 执行可视化 ---
# 检查路径是否有效
if 'val_image_dir' in locals() and os.path.isdir(val_image_dir) and \
        'output_mask_dir' in locals() and os.path.isdir(output_mask_dir):

    # 查找所有原始验证图像 (例如 .png, 根据你的文件类型修改)
    val_image_paths = sorted(glob.glob(os.path.join(val_image_dir, "*.png")))

    if not val_image_paths:
        print(f"错误：在目录 {val_image_dir} 中未找到任何 .png 图像文件。")
    else:
        print(f"找到 {len(val_image_paths)} 张验证图像，开始可视化...")

        # 遍历所有找到的验证图像
        for img_path in tqdm(val_image_paths, desc="可视化进度"):
            # 构建对应的掩码文件路径
            base_name = os.path.basename(img_path)
            mask_path = os.path.join(output_mask_dir, base_name) # 假设掩码文件名与原图名一致
            origin_output_mask_path = os.path.join(output_mask_origin_output_dir, base_name) # 原始输出掩码路径
            save_path = os.path.join(output_dir, base_name) # 保存路径

            # 检查对应的掩码文件是否存在
            if os.path.exists(mask_path):
                # 调用可视化函数，包含第二个掩码和保存路径
                visualize_overlay(img_path, mask_path, origin_output_mask_path, save_path)
            else:
                print(f"警告：找不到对应的预测掩码文件 {mask_path}，跳过图像 {base_name}。")

        print("\n所有验证结果可视化完成。")
else:
    print("错误：无法执行可视化，因为 'val_image_dir' 或 'output_mask_dir' 无效。")