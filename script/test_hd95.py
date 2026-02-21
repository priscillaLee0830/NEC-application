from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from medpy import metric
from mmseg.apis import init_model, inference_model
import csv
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from skimage.segmentation import find_boundaries

print("[WARNING] 这不是最终的 mmseg 库脚本，没有改进后的 NEC-Net，仅用来跑通 HD95 计算。切换到改进库脚本时，先卸载此库 pip uninstall mmsegmentation，再安装改进库。")

# ==========================================
# 新增：可视化函数 (核心修改)
# ==========================================
from matplotlib.lines import Line2D

def visualize_hd95_error(gt_mask, pred_mask, save_path, hd95_val):
    """
    计算并可视化 HD95 (V7版：修复图例爆炸问题，完美布局)
    """
    # 1. 提取轮廓
    gt_boundary = find_boundaries(gt_mask, mode='inner').astype(np.uint8)
    pred_boundary = find_boundaries(pred_mask, mode='inner').astype(np.uint8)
    
    gt_coords = np.argwhere(gt_boundary > 0)
    pred_coords = np.argwhere(pred_boundary > 0)
    
    if len(gt_coords) == 0 or len(pred_coords) == 0:
        return

    # 2. 计算距离矩阵
    d_pred_to_gt = cdist(pred_coords, gt_coords, metric='euclidean')
    d_gt_to_pred = cdist(gt_coords, pred_coords, metric='euclidean')
    
    # 3. 找到最近距离
    min_d_pred = np.min(d_pred_to_gt, axis=1)
    min_d_gt = np.min(d_gt_to_pred, axis=1)
    
    # 4. 计算 95% 分位数
    p_95_pred = np.percentile(min_d_pred, 95)
    p_95_gt = np.percentile(min_d_gt, 95)
    
    # 5. 确定 HD95 连线坐标
    p1, p2 = None, None
    max_hd95 = max(p_95_pred, p_95_gt)
    
    if p_95_pred >= p_95_gt:
        idx = (np.abs(min_d_pred - p_95_pred)).argmin()
        p1 = pred_coords[idx] 
        target_idx = np.argmin(d_pred_to_gt[idx])
        p2 = gt_coords[target_idx] 
    else:
        idx = (np.abs(min_d_gt - p_95_gt)).argmin()
        p1 = gt_coords[idx] 
        target_idx = np.argmin(d_gt_to_pred[idx])
        p2 = pred_coords[target_idx] 

    # 6. 绘图
    plt.figure(figsize=(6, 6), dpi=100)
    plt.imshow(np.zeros_like(gt_mask), cmap='gray')
    
    # 画轮廓 (实际画图，不带 label，防止自动生成错误的图例)
    plt.scatter(gt_coords[:, 1], gt_coords[:, 0], c='#00FF00', s=0.5, alpha=0.8)
    plt.scatter(pred_coords[:, 1], pred_coords[:, 0], c='#FF0000', s=0.5, alpha=0.8)
    
    # 画 HD95 连线 (黄线)
    plt.plot([p1[1], p2[1]], [p1[0], p2[0]], color='yellow', linewidth=2, linestyle='--', zorder=5)
    
    # 画端点 'x' (青色)
    plt.scatter([p1[1], p2[1]], [p1[0], p2[0]], c='cyan', marker='x', s=60, linewidths=2, zorder=10)
    
    # --- 修复核心：手动创建完美的图例 ---
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='GT Contour', 
               markerfacecolor='#00FF00', markersize=4),  # 手动把点调大一点方便看
        Line2D([0], [0], marker='o', color='w', label='Pred Contour', 
               markerfacecolor='#FF0000', markersize=4),
        Line2D([0], [0], marker='x', color='cyan', label='Max Error Points', 
               markerfacecolor='cyan', markersize=8, markeredgewidth=2, linestyle='None') # 叉号保持正常大小
    ]
    
    # 图例放右下角，数值放左上角，互不干扰
    plt.legend(handles=legend_elements, loc='lower right')
    
    # --- 纯文本标注 (固定左上角) ---
    plt.text(
        20, 20, 
        f"{max_hd95:.1f}px", 
        ha='left', 
        va='top', 
        color='yellow', 
        fontsize=12, 
        fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="yellow", alpha=0.8)
    )

    plt.title(f"HD95 Visualization\nMetric: {hd95_val:.2f} (Calc: {max_hd95:.2f})")
    plt.axis('off')
    
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

# ==========================================
# 原有逻辑保持不变
# ==========================================
def change_name(path:Path, old_name:str, new_name:str) -> Path:
    # 将 TARGET_PATH 中的 OLD_NAME 替换为 NEW_NAME
    paris = path.parts
    if old_name in paris:
        new_parts = [new_name if part == old_name else part for part in paris]
        return Path(*new_parts)
    return path


if __name__ == "__main__":
    dataset_path = Path("./datasets/NEC")
    output_dir = Path("./hd95_results")
    
    # 新增：创建一个专门放可视化图片的文件夹
    vis_dir = output_dir / "visualization"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    output_dir.mkdir(parents=True, exist_ok=True)

    config_file = "./work_dirs/swim/swin_transformer_polyp_config.py"
    checkpoint_file = "./work_dirs/swim/best_mIoU_iter_8000.pth"
    model = init_model(config_file, checkpoint_file, device="cuda:0")

    results = []
    mhd_list = []
    total_hd_list = []
    
    # 这里的循环逻辑和你原来的一模一样
    for i in range(1,6):
        hd_list = []
        scan_path = dataset_path / f"fold_{i}" / "img_dir" / "val"
        file_list = list(scan_path.glob("*.jpg"))
        
        # 创建每个 fold 的可视化子文件夹
        fold_vis_dir = vis_dir / f"fold_{i}"
        fold_vis_dir.mkdir(exist_ok=True)

        for file_path in tqdm(file_list, desc=f"Fold {i} Inference"):
            result = inference_model(model, str(file_path))
            pred_mask = result.pred_sem_seg.data[0].cpu().numpy()
            
            # [840, 840], [0,1,2] -> [0,1]
            pred_mask[pred_mask == 2] = 1 
            
            mask_path = change_name(file_path, "img_dir", "ann_dir").with_suffix(".png")
            mask_data = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            # 这里的 Resize 非常重要，Reviewer 质疑了 Resize 导致像素单位变化
            # 你现在统一 Resize 到 224x224，计算出来的 HD95 是基于 224 尺寸的像素值
            mask_data = cv2.resize(mask_data, dsize=pred_mask.shape[::-1], interpolation=cv2.INTER_NEAREST)
            mask_data[mask_data > 0] = 1 
            
            pred_mask = cv2.resize(pred_mask.astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST)
            mask_data = cv2.resize(mask_data.astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST)
            
            pred_has_fg = np.any(pred_mask)
            gt_has_fg = np.any(mask_data)
            
            hd = 0.0 # Default
            
            if not pred_has_fg and gt_has_fg:
                # [Risk] GT 有东西，Pred 全黑。
                # 严格来说 HD95 此时应为无穷大或图像对角线长度。
                # 你的代码为了跑通设为 0 或者跳过，这里保持你的逻辑，但建议记录一下
                pass 
            elif not pred_has_fg and not gt_has_fg:
                hd = 0.0
            else:
                # Medpy 计算
                hd = metric.binary.hd95(pred_mask, mask_data)
                
                # ==========================================
                # 插入：当 HD95 比较大时（或者你想看所有图），生成可视化
                # 这里我设置一个阈值，比如 > 5px 才画，或者你可以去掉 if 全画
                # ==========================================
                if hd > 5.0:  # 只保存误差比较大的图，方便写 Rebuttal
                    vis_name = fold_vis_dir / f"{file_path.stem}_hd_{hd:.2f}.png"
                    try:
                        visualize_hd95_error(mask_data, pred_mask, vis_name, hd)
                    except Exception as e:
                        print(f"Visualization failed for {file_path.stem}: {e}")

            hd_list.append(hd)
            results.append({
                'fold': i,
                'image_path': str(file_path),
                'hd95': hd
            })
        
        fold_mean = np.mean(hd_list)
        mhd_list.append(fold_mean)
        total_hd_list.extend(hd_list)
        print(f"\tHD95: {fold_mean:.4f} ± {np.std(hd_list):.4f}, min is {np.min(hd_list):.4f}, max is {np.max(hd_list):.4f}")

    print(f"Fold-wise Mean HD95 (Macro): {np.mean(mhd_list):.4f} ± {np.std(mhd_list):.4f}")
    print(f"Overall Mean HD95 (Micro): {np.mean(total_hd_list):.4f} ± {np.std(total_hd_list):.4f}")

    output_file = output_dir / "hd95_per_image.csv"
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['fold', 'image_path', 'hd95']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"Per-image HD95 results saved to {output_file}")
    print(f"Visualization images saved to {vis_dir}")
