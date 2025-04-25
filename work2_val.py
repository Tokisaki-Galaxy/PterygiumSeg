# %%
import os
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import zipfile
import tempfile
import shutil


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 配置路径 ---
val_image_dir=      r'f:\val_img'
output_mask_zip = 'mask.zip'
output_mask_origin_zip = 'mask_original.zip'
output_dir = r'output'

# 创建输出目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)

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

# --- 解压函数 ---
def extract_zip(zip_path, extract_to):
    """解压 zip 文件到指定目录"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            print(f"正在解压 {zip_path} 到 {extract_to}...")
            zip_ref.extractall(extract_to)
            print(f"解压完成。")
        return True
    except FileNotFoundError:
        print(f"错误: 找不到 zip 文件 {zip_path}")
        return False
    except Exception as e:
        print(f"解压 {zip_path} 时出错: {e}")
        return False

# --- 执行可视化 ---
if 'val_image_dir' in locals() and os.path.isdir(val_image_dir) and \
    os.path.isfile(output_mask_zip) and os.path.isfile(output_mask_origin_zip):

    # 创建临时目录来解压文件
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_mask_dir = os.path.join(temp_dir, "extracted_masks")
        temp_mask_origin_dir = os.path.join(temp_dir, "extracted_masks_origin")

        # 解压掩码文件
        masks_extracted = extract_zip(output_mask_zip, temp_mask_dir)
        origin_masks_extracted = extract_zip(output_mask_origin_zip, temp_mask_origin_dir)

        if not masks_extracted or not origin_masks_extracted:
            print("错误：无法解压必要的掩码文件，可视化中止。")
        else:
            # 查找所有原始验证图像 (例如 .png, 根据你的文件类型修改)
            val_image_paths = sorted(glob.glob(os.path.join(val_image_dir, "*.png")))

            if not val_image_paths:
                print(f"错误：在目录 {val_image_dir} 中未找到任何 .png 图像文件。")
            else:
                print(f"找到 {len(val_image_paths)} 张验证图像，开始可视化...")

                # 遍历所有找到的验证图像
                for img_path in tqdm(val_image_paths, desc="可视化进度"):
                    # 构建对应的掩码文件路径 (在临时目录中查找)
                    base_name = os.path.basename(img_path)
                    mask_path = os.path.join(temp_mask_dir, base_name) # 在解压后的目录查找
                    origin_output_mask_path = os.path.join(temp_mask_origin_dir, base_name) # 在解压后的目录查找
                    save_path = os.path.join(output_dir, base_name) # 保存路径

                    # 检查对应的掩码文件是否存在
                    if os.path.exists(mask_path):
                        # 调用可视化函数，包含第二个掩码和保存路径
                        visualize_overlay(img_path, mask_path, origin_output_mask_path, save_path)
                    else:
                        print(f"警告：在解压目录中找不到对应的预测掩码文件 {mask_path}，跳过图像 {base_name}。")

                print("\n所有验证结果可视化完成。")

else:
    error_msg = "错误：无法执行可视化，因为 "
    if 'val_image_dir' not in locals() or not os.path.isdir(val_image_dir):
        error_msg += "'val_image_dir' 无效；"
    if not os.path.isfile(output_mask_zip):
        error_msg += f"掩码 zip 文件 '{output_mask_zip}' 不存在；"
    if not os.path.isfile(output_mask_origin_zip):
        error_msg += f"原始掩码 zip 文件 '{output_mask_origin_zip}' 不存在；"
    print(error_msg.strip('；'))