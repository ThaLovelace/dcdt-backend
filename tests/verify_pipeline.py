"""
tests/verify_pipeline.py
------------------------
สคริปต์ทดสอบ Sanity Check สำหรับเทียบความถูกต้องของ Backend กับ Training
python tests/verify_pipeline.py

"""

import os
import sys
import base64
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. จัดการ Path ให้มองเห็นโฟลเดอร์ core
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
sys.path.insert(0, str(root_dir))

from core.vision_bridge import prepare_image_for_vit
from core.inference import run_real_vit_inference

def run_thesis_visual_test():
    # ⚠️ ชื่อไฟล์รูปดิบ (Raw Image) ที่ยังไม่ได้ทำอะไรเลย
    image_name = "test_image.png" 
    image_path = current_dir / image_name
    
    if not image_path.exists():
        print(f"❌ ไม่พบไฟล์รูปภาพ: {image_path}")
        return

    print("==================================================")
    print("🚀 เริ่มการทดสอบ Backend Pipeline & สร้างรูปภาพ")
    print("==================================================")

    # โหลดรูปดิบมาเก็บไว้สำหรับวาดกราฟ (Figure 1 ฝั่งซ้าย)
    raw_img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    # จำลองการส่งข้อมูลแบบ Base64 จากหน้าบ้าน
    with open(image_path, "rb") as f:
        img_b64 = f"data:image/png;base64,{base64.b64encode(f.read()).decode('utf-8')}"

    print("⏳ กำลังประมวลผล Vision Bridge และ XAI Engine...")

    # ---------------------------------------------------------
    # STEP A: ทดสอบ Vision Bridge (ดึงค่า Tensor Sum)
    # ---------------------------------------------------------
    input_tensor, img_pil = prepare_image_for_vit(img_b64)
    t_sum = input_tensor.sum().item()
    
    # แปลงภาพ PIL (3px) กลับเป็น Grayscale เพื่อเอาไปวาดกราฟ (Figure 1 ฝั่งขวา)
    bridge_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2GRAY)

    # ---------------------------------------------------------
    # STEP B: ทดสอบ Inference (จำลองรันทั้งระบบแบบสมบูรณ์)
    # ---------------------------------------------------------
    # ฟังก์ชันนี้จะรับรูปดิบ (Base64) ไปแปลง 3px และทำ XAI ให้เองตามลำดับที่ถูกต้อง
    is_risk, conf, heatmap_b64 = run_real_vit_inference(img_b64)
    pred_label = "Risk (1)" if is_risk else "Normal (0)"

    print(f"👉 [BACKEND TENSOR SUM] : {t_sum:.6f}")
    print(f"👉 [BACKEND CONFIDENCE] : {conf:.2f}% (Class: {pred_label})")

    # ถอดรหัสรูป Heatmap กลับมาเพื่อใช้วาดกราฟ
    heatmap_data = base64.b64decode(heatmap_b64.split(",")[1])
    heatmap_np = np.frombuffer(heatmap_data, np.uint8)
    heatmap_img = cv2.imdecode(heatmap_np, cv2.IMREAD_COLOR)

    # ---------------------------------------------------------
    # STEP C: สร้างรูปภาพสำหรับวิทยานิพนธ์ (เหมือน Colab)
    # ---------------------------------------------------------
    print("📊 กำลังสร้างรูปภาพกราฟิก...")

    # --- FIGURE 1: Digital Bridge (เทียบรูปดิบ กับ รูป 3px) ---
    fig1, ax1 = plt.subplots(1, 2, figsize=(10, 5), dpi=150)
    fig1.patch.set_facecolor('white')
    ax1[0].imshow(raw_img, cmap='gray')
    ax1[0].set_title("Original Raw Scan", fontsize=12, pad=10)
    ax1[1].imshow(bridge_img, cmap='gray')
    ax1[1].set_title(f"After Vision Bridge (3px)\n[Tensor Sum: {t_sum:.6f}]", fontsize=12, pad=10)
    for a in ax1: a.axis('off')
    
    # เซฟรูปที่ 1 และแสดงผล
    out_fig1 = current_dir / "backend_report_fig1_bridge.png"
    plt.tight_layout()
    plt.savefig(out_fig1, bbox_inches="tight", facecolor='white')
    plt.show()

    # --- FIGURE 2: XAI Decision (เทียบรูป 3px ต้นฉบับ กับ รูป Heatmap) ---
    fig2, ax2 = plt.subplots(1, 2, figsize=(10, 5), dpi=150)
    fig2.patch.set_facecolor('white')
    # ใช้รูป 3px ที่ Resize แล้วมาโชว์เทียบ
    ax2[0].imshow(np.array(img_pil.resize((224, 224))))
    ax2[0].set_title(f"Input Image\nPred: {pred_label} ({conf:.2f}%)", fontsize=12, fontweight='bold', pad=10)
    
    ax2[1].imshow(cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB))
    ax2[1].set_title(f"Decision Heatmap (Chefer)\nPred: {pred_label} ({conf:.2f}%)", fontsize=12, fontweight='bold', color='darkred', pad=10)
    for a in ax2: a.axis('off')

    # เซฟรูปที่ 2 และแสดงผล
    out_fig2 = current_dir / "backend_report_fig2_xai.png"
    plt.tight_layout()
    plt.savefig(out_fig2, bbox_inches="tight", facecolor='white')
    plt.show()

    print(f"\n✅ สร้างและบันทึกรูปภาพสำเร็จ! ไฟล์อยู่ในโฟลเดอร์: {current_dir.name}")
    print("==================================================")

if __name__ == "__main__":
    run_thesis_visual_test()