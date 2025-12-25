import os
import numpy as np
import cv2
import yaml
from tqdm import tqdm
from collections import defaultdict

# 配置
YOLO_CONFIG = "aortic_valve_colab.yaml"
TEST_IMAGES_DIR = "datasets/test/images"
PREDICTIONS_DIR = "run/detect/val/labels"
OUTPUT_DIR = "evaluation_results"
IOU_THRESHOLD = 0.5  # 用於匹配預測和真實框的 IoU 閾值
CONFIDENCE_THRESHOLD = 0.25  # 考慮預測的最小置信度

# 創建輸出目錄
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/visualizations", exist_ok=True)

# 從 YAML 加載類別名稱
with open(YOLO_CONFIG, 'r') as f:
    yaml_data = yaml.safe_load(f)
class_names = yaml_data['names']

def read_yolo_label_file(label_path, img_width=640, img_height=640):
    """讀取 YOLO 格式的標籤文件，返回邊界框列表 [x1, y1, x2, y2, conf, class_id]"""
    if not os.path.exists(label_path):
        print(f"警告: 標籤文件不存在: {label_path}")
        return []
    
    boxes = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                    
                try:
                    # 處理類別 ID
                    class_id = int(float(parts[0]))
                    x_center, y_center, width, height = map(float, parts[1:5])
                    conf = float(parts[5]) if len(parts) > 5 else 1.0
                    
                    # 從 YOLO 格式轉換為 [x1, y1, x2, y2]
                    x1 = max(0, int((x_center - width/2) * img_width))
                    y1 = max(0, int((y_center - height/2) * img_height))
                    x2 = min(img_width, int((x_center + width/2) * img_width))
                    y2 = min(img_height, int((y_center + height/2) * img_height))
                    
                    # 確保邊界框有效
                    if x1 < x2 and y1 < y2:
                        boxes.append([x1, y1, x2, y2, conf, class_id])
                    else:
                        print(f"警告: 跳過無效的邊界框: {[x1, y1, x2, y2]} 在 {label_path}")
                        
                except (ValueError, IndexError) as e:
                    print(f"警告: 無法解析行: {line.strip()} 在 {label_path}. 錯誤: {e}")
                    continue
    except Exception as e:
        print(f"讀取標籤文件時出錯 {label_path}: {e}")
    
    return boxes

def calculate_iou(box1, box2):
    """計算兩個邊界框之間的交並比 (IoU)"""
    # 計算交集區域
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # 計算交集面積
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # 計算並集面積
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    # 計算 IoU
    iou = inter_area / union_area if union_area > 0 else 0.0
    
    return iou

def draw_dashed_rect(image, pt1, pt2, color, thickness=1, dash_length=5):
    """繪製虛線矩形"""
    x1, y1 = pt1
    x2, y2 = pt2
    
    # 繪製四條邊
    draw_dashed_line(image, (x1, y1), (x2, y1), color, thickness, dash_length)  # 上邊
    draw_dashed_line(image, (x2, y1), (x2, y2), color, thickness, dash_length)  # 右邊
    draw_dashed_line(image, (x2, y2), (x1, y2), color, thickness, dash_length)  # 下邊
    draw_dashed_line(image, (x1, y2), (x1, y1), color, thickness, dash_length)  # 左邊

def draw_dashed_line(image, pt1, pt2, color, thickness=1, dash_length=5):
    """繪製虛線"""
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
    dashes = int(dist / dash_length) or 1  # 避免除以零
    
    for i in range(0, dashes):
        start = [int(pt1[0] + (pt2[0] - pt1[0]) * i / dashes),
                 int(pt1[1] + (pt2[1] - pt1[1]) * i / dashes)]
        end = [int(pt1[0] + (pt2[0] - pt1[0]) * (i + 0.5) / dashes),
               int(pt1[1] + (pt2[1] - pt1[1]) * (i + 0.5) / dashes)]
        cv2.line(image, tuple(start), tuple(end), color, thickness)

def evaluate_predictions():
    """評估預測結果"""
    # 初始化指標
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_iou = 0
    results = []
    
    # 獲取測試圖片列表
    test_images = [f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not test_images:
        print(f"錯誤: 在 {TEST_IMAGES_DIR} 中找不到測試圖片")
        return
    
    print(f"找到 {len(test_images)} 張測試圖片")
    
    for img_name in tqdm(test_images, desc="處理圖片"):
        try:
            # 構建路徑
            img_path = os.path.join(TEST_IMAGES_DIR, img_name)
            label_path = os.path.join(TEST_IMAGES_DIR.replace('images', 'labels'), 
                                    os.path.splitext(img_name)[0] + '.txt')
            pred_path = os.path.join(PREDICTIONS_DIR, os.path.splitext(img_name)[0] + '.txt')
            
            # 讀取圖片獲取尺寸
            img = cv2.imread(img_path)
            if img is None:
                print(f"警告: 無法讀取圖片 {img_path}")
                continue
                
            img_h, img_w = img.shape[:2]
            
            # 讀取真實標籤和預測結果
            gt_boxes = read_yolo_label_file(label_path, img_w, img_h)
            pred_boxes = read_yolo_label_file(pred_path, img_w, img_h)
            pred_boxes = [box for box in pred_boxes if box[4] >= CONFIDENCE_THRESHOLD]
            
            # 初始化變量
            tp = 0
            fp = 0
            fn = 0
            matched_gt = set()
            matched_pred = set()
            iou_sum = 0
            
            # 將預測與真實標籤匹配
            for i, gt_box in enumerate(gt_boxes):
                best_iou = IOU_THRESHOLD
                best_match = -1
                
                for j, pred_box in enumerate(pred_boxes):
                    if j in matched_pred:
                        continue
                        
                    iou = calculate_iou(gt_box[:4], pred_box[:4])
                    if iou > best_iou:
                        best_iou = iou
                        best_match = j
                
                if best_match != -1:
                    tp += 1
                    matched_gt.add(i)
                    matched_pred.add(best_match)
                    iou_sum += best_iou
            
            # 計算 FP 和 FN
            fp = len(pred_boxes) - len(matched_pred)
            fn = len(gt_boxes) - len(matched_gt)
            
            # 更新總數
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_iou += iou_sum if tp > 0 else 0
            
            # 創建可視化
            if gt_boxes or pred_boxes:  # 只有當有框時才可視化
                img_viz = img.copy()
                img_viz = cv2.cvtColor(img_viz, cv2.COLOR_BGR2RGB)  # 轉換為 RGB 用於顯示
                
                # 繪製真實標籤（綠色）
                for i, box in enumerate(gt_boxes):
                    color = (0, 255, 0)  # 綠色表示真實標籤
                    if i in matched_gt:
                        cv2.rectangle(img_viz, (box[0], box[1]), (box[2], box[3]), color, 2)
                        cv2.putText(img_viz, f"GT {class_names.get(box[5], box[5])}", 
                                  (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    else:
                        # 使用虛線繪製未檢測到的真實標籤（假陰性）
                        draw_dashed_rect(img_viz, (box[0], box[1]), (box[2], box[3]), color, 2, dash_length=5)
                        cv2.putText(img_viz, f"FN {class_names.get(box[5], box[5])}", 
                                  (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # 繪製預測結果（紅色）
                for j, box in enumerate(pred_boxes):
                    color = (255, 0, 0)  # 紅色表示預測結果
                    if j in matched_pred:
                        cv2.rectangle(img_viz, (box[0], box[1]), (box[2], box[3]), color, 1)
                        cv2.putText(img_viz, f"Pred {class_names.get(box[5], box[5])} {box[4]:.2f}", 
                                  (box[0], box[3]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    else:
                        # 使用虛線繪製錯誤的預測（假陽性）
                        draw_dashed_rect(img_viz, (box[0], box[1]), (box[2], box[3]), color, 1, dash_length=3)
                        cv2.putText(img_viz, f"FP {class_names.get(box[5], box[5])} {box[4]:.2f}", 
                                  (box[0], box[3]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 0, 0), 1)
                
                # 保存可視化結果
                viz_path = os.path.join(OUTPUT_DIR, "visualizations", img_name)
                cv2.imwrite(viz_path, cv2.cvtColor(img_viz, cv2.COLOR_RGB2BGR))
            
            # 存儲結果
            results.append({
                'image': img_name,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'avg_iou': iou_sum / tp if tp > 0 else 0
            })
            
        except Exception as e:
            print(f"處理圖片 {img_name} 時出錯: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 計算總體指標
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    mean_iou = total_iou / total_tp if total_tp > 0 else 0
    
    # 生成報告
    generate_report(results, precision, recall, f1_score, mean_iou, len(test_images))

def generate_report(results, precision, recall, f1_score, mean_iou, total_images):
    """生成評估報告"""
    report_path = os.path.join(OUTPUT_DIR, "evaluation_report.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        # 寫入標題
        f.write("# YOLO 物件偵測評估報告\n\n")
        
        # 摘要指標
        f.write("## 摘要指標\n")
        f.write(f"- **精確率 (Precision)**: {precision:.4f}\n")
        f.write(f"- **召回率 (Recall)**: {recall:.4f}\n")
        f.write(f"- **F1 分數**: {f1_score:.4f}\n")
        f.write(f"- **平均 IoU**: {mean_iou:.4f}\n")
        f.write(f"- **總圖片數**: {total_images}\n")
        f.write(f"- **真正例 (TP)**: {sum(r['tp'] for r in results)}\n")
        f.write(f"- **假正例 (FP)**: {sum(r['fp'] for r in results)}\n")
        f.write(f"- **假負例 (FN)**: {sum(r['fn'] for r in results)}\n\n")
        
        # 每張圖片的結果
        f.write("## 每張圖片的結果\n")
        f.write("| 圖片 | TP | FP | FN | 精確率 | 召回率 | mIoU |\n")
        f.write("|------|----|----|----|--------|--------|------|\n")
        
        for r in results:
            f.write(f"| {r['image']} | {r['tp']} | {r['fp']} | {r['fn']} | {r['precision']:.2f} | {r['recall']:.2f} | {r['avg_iou']:.2f} |\n")
        
        # 詳細分析
        f.write("\n## 詳細分析\n")
        
        # 假正例最多的圖片
        fp_images = [r for r in results if r['fp'] > 0]
        if fp_images:
            fp_images = sorted(fp_images, key=lambda x: x['fp'], reverse=True)[:5]
            f.write("\n### 假正例最多的 5 張圖片\n")
            for img in fp_images:
                f.write(f"- `{img['image']}`: {img['fp']} 個假正例\n")
        else:
            f.write("\n### 沒有發現假正例\n")
        
        # 假負例最多的圖片
        fn_images = [r for r in results if r['fn'] > 0]
        if fn_images:
            fn_images = sorted(fn_images, key=lambda x: x['fn'], reverse=True)[:5]
            f.write("\n### 假負例最多的 5 張圖片\n")
            for img in fn_images:
                f.write(f"- `{img['image']}`: {img['fn']} 個假負例\n")
        else:
            f.write("\n### 沒有發現假負例\n")
        
        # 可視化說明
        f.write("\n## 可視化說明\n")
        f.write("比較真實標籤（綠色）和預測結果（紅色）的可視化結果已保存在 `evaluation_results/visualizations` 目錄中。\n")
        f.write("- **實線綠框**: 正確檢測到的物體\n")
        f.write("- **虛線綠框**: 未檢測到的物體（假負例）\n")
        f.write("- **實線紅框**: 正確的預測\n")
        f.write("- **虛線紅框**: 錯誤的預測（假正例）\n")
    
    print(f"\n評估完成。報告已保存至: {os.path.abspath(report_path)}")
    print(f"可視化結果已保存至: {os.path.abspath(os.path.join(OUTPUT_DIR, 'visualizations'))}")

if __name__ == "__main__":
    try:
        print("開始評估...")
        print(f"當前工作目錄: {os.getcwd()}")
        print(f"測試圖片目錄: {os.path.abspath(TEST_IMAGES_DIR)}")
        print(f"預測結果目錄: {os.path.abspath(PREDICTIONS_DIR)}")
        
        # 檢查目錄是否存在
        if not os.path.exists(TEST_IMAGES_DIR):
            print(f"錯誤: 測試圖片目錄不存在: {os.path.abspath(TEST_IMAGES_DIR)}")
        if not os.path.exists(PREDICTIONS_DIR):
            print(f"錯誤: 預測結果目錄不存在: {os.path.abspath(PREDICTIONS_DIR)}")
            print("請確保您已經運行了 YOLO 檢測以生成預測結果。")
            
        # 檢查目錄中是否有文件
        test_images = [f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        pred_files = [f for f in os.listdir(PREDICTIONS_DIR) if f.endswith('.txt')]
        
        print(f"找到 {len(test_images)} 張測試圖片和 {len(pred_files)} 個預測文件")
        
        if test_images:
            if pred_files:
                evaluate_predictions()
            else:
                print(f"錯誤: 在 {os.path.abspath(PREDICTIONS_DIR)} 中找不到預測文件")
        else:
            print(f"錯誤: 在 {os.path.abspath(TEST_IMAGES_DIR)} 中找不到測試圖片")
            
    except Exception as e:
        print(f"發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
        input("按 Enter 鍵退出...")