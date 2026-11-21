# =============================================================================
#  EVALUASI YOLO12n vs YOLO26n  (Botol Beling, Botol Plastik, Kaleng)
#  - TP/TN/FP/FN  : level GAMBAR, rumus ONE-vs-REST per kelas
#  - Precision, Recall, F1, Accuracy : dari TP/TN/FP/FN one-vs-rest
#  - Mean IoU     : rata-rata IoU dari deteksi TP
#  - mAP@0.5      : level OBJEK (kurva PR, semua prediksi conf >= 0.001)
#  Output : log (.log) + gambar anotasi + heatmap confusion matrix (.png)
# =============================================================================
#
#  RUMUS (one-vs-rest, untuk kelas c, N = jumlah gambar):
#    TP_c : GT = c  DAN  prediksi = c  DAN  IoU >= 0.5
#    FN_c : GT = c  tetapi TP_c tidak terpenuhi
#           (tidak terdeteksi / salah kelas / IoU < 0.5)
#    FP_c : model memprediksi c, padahal GT bukan c
#    TN_c : N - TP_c - FP_c - FN_c   (semua gambar sisanya)
#
#    Precision = TP / (TP + FP)
#    Recall    = TP / (TP + FN)
#    F1        = 2 * P * R / (P + R)
#    Accuracy  = (TP + TN) / (TP + TN + FP + FN)
#    Mean IoU  = sum(IoU deteksi TP) / jumlah TP
#    AP@0.5    = luas di bawah kurva Precision-Recall (IoU 0.5)
#    mAP@0.5   = rata-rata AP@0.5 seluruh kelas
#
#  Baris TOTAL = penjumlahan TP/TN/FP/FN seluruh kelas (micro-average).
# =============================================================================

import os
import re
import glob
import shutil
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np
import pandas as pd

# =============================================================================
# KONFIGURASI
# =============================================================================
BASE_DIR      = r"C:\Users\acer\Downloads\Tugas Akhir Finals\dataset"
SCENARIO_NAME = "Skenario A"            # ganti sesuai skenario yang diuji
DATASET       = os.path.join(BASE_DIR, "uji-hasil")
IMG_DIR       = os.path.join(DATASET, "images")
GT_DIR        = os.path.join(DATASET, "labels")

LOG_OUT = os.path.join(BASE_DIR, f"evaluation_log_{SCENARIO_NAME.replace(' ', '_')}.log")

MODELS = {
    "YOLO12n": os.path.join(BASE_DIR, "YOLO12n.pt"),
    "YOLO26n": os.path.join(BASE_DIR, "YOLO26n.pt"),
}

CLASS_NAMES = {0: "Botol Beling", 1: "Botol Plastik", 2: "Kaleng"}

IOU_THRESHOLD  = 0.5     # batas IoU untuk TP dan AP@0.5
CONF_THRESHOLD = 0.25    # batas confidence untuk TP/TN/FP/FN & gambar anotasi
MAP_CONF       = 0.001   # confidence sangat rendah khusus untuk menghitung mAP
NMS_IOU        = 0.45
IMAGE_SIZE     = 640

LIGHT_ORDER = ["Sedang", "Terang", "Redup"]
IMG_EXTS    = (".jpg", ".jpeg", ".png", ".bmp")

BG_COL = "Tidak Terdeteksi"   # kolom confusion matrix: tidak ada prediksi valid
BG_ROW = "Tidak Ada Objek"    # baris confusion matrix: gambar tanpa GT


# =============================================================================
# UTILITAS
# =============================================================================
def natural_sort_key(text: str):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", text)]


def build_image_map():
    """stem -> path gambar (mendukung beberapa ekstensi)."""
    m = {}
    for p in glob.glob(os.path.join(IMG_DIR, "*")):
        stem, ext = os.path.splitext(os.path.basename(p))
        if ext.lower() in IMG_EXTS:
            m[stem] = p
    return m


def get_lighting(img_path: str, img_name: str) -> str:
    n = img_name.lower()
    if "_r." in n or "_redup" in n:
        return "Redup"
    if "_t." in n or "_terang" in n:
        return "Terang"
    if "_s." in n or "_sedang" in n:
        return "Sedang"
    # cadangan: tebak dari kecerahan rata-rata
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) if img_path else None
    b = float(img.mean()) if img is not None else 128.0
    return "Redup" if b < 85 else ("Sedang" if b < 170 else "Terang")


def load_yolo_annotations(path: str):
    """Return list (cls, x, y, w, h, conf). GT tidak punya conf -> 1.0"""
    if not os.path.exists(path):
        return []
    boxes = []
    with open(path, "r") as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 5:
                continue
            conf = float(p[5]) if len(p) >= 6 else 1.0
            boxes.append((int(float(p[0])), *map(float, p[1:5]), conf))
    return boxes


def xywhn_to_xyxy(b):
    _, x, y, w, h = b[:5]
    return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)


def compute_iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def safe_div(a, b):
    return a / b if b > 0 else 0.0


# =============================================================================
# mAP@0.5 (level objek)
# =============================================================================
def calculate_ap(recall, precision):
    if len(recall) == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def compute_ap50(gt_data, pred_data, class_ids, thr=IOU_THRESHOLD):
    """AP@thr tiap kelas. Kelas tanpa GT -> NaN (tidak ikut rata-rata)."""
    result = {}
    for c in class_ids:
        gts = defaultdict(list)
        n_gt = 0
        for g in gt_data:
            if g["cls"] == c:
                gts[g["img"]].append(g["xyxy"])
                n_gt += 1
        if n_gt == 0:
            result[c] = float("nan")
            continue

        preds = sorted((p for p in pred_data if p["cls"] == c),
                       key=lambda x: x["conf"], reverse=True)
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        used = defaultdict(set)

        for i, p in enumerate(preds):
            best_iou, best_j = 0.0, -1
            for j, gbox in enumerate(gts[p["img"]]):
                if j in used[p["img"]]:
                    continue
                iou = compute_iou(p["xyxy"], gbox)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j >= 0 and best_iou >= thr:
                tp[i] = 1
                used[p["img"]].add(best_j)
            else:
                fp[i] = 1

        tp_c, fp_c = np.cumsum(tp), np.cumsum(fp)
        rec = tp_c / (n_gt + 1e-16)
        prec = tp_c / (tp_c + fp_c + 1e-16)
        result[c] = calculate_ap(rec, prec)
    return result


def mean_ignore_nan(values):
    v = [x for x in values if not np.isnan(x)]
    return float(np.mean(v)) if v else 0.0


# =============================================================================
# INFERENSI
# =============================================================================
def run_inference(model_path: str, model_name: str):
    """
    Pass 1 : conf=0.001, simpan label (txt+conf)  -> dipakai untuk metrik & mAP
    Pass 2 : conf=0.25,  simpan gambar anotasi     -> untuk dokumentasi
    """
    from ultralytics import YOLO

    project_dir = os.path.join(BASE_DIR, f"predictions_{SCENARIO_NAME.replace(' ', '_')}_{model_name}")
    if os.path.exists(project_dir):
        shutil.rmtree(project_dir)
    os.makedirs(project_dir, exist_ok=True)

    model = YOLO(model_path)
    print(f"\n[{model_name}] Inferensi pass 1 (label, conf={MAP_CONF}) ...")
    model.predict(source=IMG_DIR, imgsz=IMAGE_SIZE, conf=MAP_CONF, iou=NMS_IOU,
                  save=False, save_txt=True, save_conf=True,
                  project=project_dir, name="label_lowconf", exist_ok=True, verbose=False)

    print(f"[{model_name}] Inferensi pass 2 (gambar, conf={CONF_THRESHOLD}) ...")
    model.predict(source=IMG_DIR, imgsz=IMAGE_SIZE, conf=CONF_THRESHOLD, iou=NMS_IOU,
                  save=True, save_txt=False, show_labels=True, show_conf=True, line_width=2,
                  project=project_dir, name="hasil_deteksi", exist_ok=True, verbose=False)

    label_dir = os.path.join(project_dir, "label_lowconf", "labels")
    img_out   = os.path.join(project_dir, "hasil_deteksi")
    return label_dir, img_out, project_dir


# =============================================================================
# EVALUASI
# =============================================================================
def evaluate(pred_dir: str, image_map: dict):
    class_ids = list(CLASS_NAMES.keys())
    gt_files  = sorted(glob.glob(os.path.join(GT_DIR, "*.txt")))

    rows, gt_all, pred_all = [], [], []
    multi_gt = 0

    for gt_path in gt_files:
        fname = os.path.basename(gt_path)
        stem  = os.path.splitext(fname)[0]
        img_path = image_map.get(stem)
        img_name = os.path.basename(img_path) if img_path else stem + ".jpg"
        light = get_lighting(img_path, img_name)

        gt_boxes   = load_yolo_annotations(gt_path)
        pred_boxes = load_yolo_annotations(os.path.join(pred_dir, fname))   # semua conf >= 0.001
        if len(gt_boxes) > 1:
            multi_gt += 1

        # ---- data mAP (level objek, semua prediksi conf rendah) ----
        for b in gt_boxes:
            gt_all.append({"img": stem, "cls": b[0], "xyxy": xywhn_to_xyxy(b), "light": light})
        for b in pred_boxes:
            pred_all.append({"img": stem, "cls": b[0], "conf": b[5],
                             "xyxy": xywhn_to_xyxy(b), "light": light})

        # ---- level gambar: 1 GT, 1 prediksi dengan conf tertinggi (>= 0.25) ----
        valid_preds = [b for b in pred_boxes if b[5] >= CONF_THRESHOLD]
        gt   = gt_boxes[0] if gt_boxes else None
        pred = max(valid_preds, key=lambda x: x[5]) if valid_preds else None

        gt_cls   = gt[0] if gt else None
        pred_cls = pred[0] if pred else None
        iou = compute_iou(xywhn_to_xyxy(gt), xywhn_to_xyxy(pred)) if (gt and pred) else 0.0
        is_tp = gt is not None and pred is not None and gt_cls == pred_cls and iou >= IOU_THRESHOLD

        if is_tp:
            status = "TP"
        elif gt is None and pred is None:
            status = "TN"
        elif gt is None:
            status = "FP (objek tidak ada)"
        elif pred is None:
            status = "FN (tidak terdeteksi)"
        elif pred_cls != gt_cls:
            status = "FN+FP (salah kelas)"
        else:
            status = "FN (IoU rendah)"

        rows.append({
            "Image Name": img_name, "Lighting": light,
            "Actual": CLASS_NAMES.get(gt_cls, BG_ROW),
            "Prediction": CLASS_NAMES.get(pred_cls, "-"),
            "gt_cls": gt_cls, "pred_cls": pred_cls, "IoU": round(iou, 4),
            "is_tp": is_tp, "Status": status,
        })

    if multi_gt:
        print(f"[PERINGATAN] {multi_gt} gambar punya >1 objek GT; level gambar hanya memakai GT pertama.")

    order = {k: i for i, k in enumerate(LIGHT_ORDER)}
    rows.sort(key=lambda r: (order.get(r["Lighting"], 99), natural_sort_key(r["Image Name"])))
    df = pd.DataFrame(rows)
    return df, gt_all, pred_all


# ---------- hitung one-vs-rest ----------
def ovr_counts(df, c):
    n  = len(df)
    tp = int(((df["gt_cls"] == c) & df["is_tp"]).sum())
    fn = int(((df["gt_cls"] == c) & ~df["is_tp"]).sum())
    fp = int(((df["pred_cls"] == c) & (df["gt_cls"] != c)).sum())
    tn = n - tp - fn - fp
    return tp, tn, fp, fn


def metric_row(tp, tn, fp, fn, sum_iou, n_iou, map50):
    prec = safe_div(tp, tp + fp)
    rec  = safe_div(tp, tp + fn)
    f1   = safe_div(2 * prec * rec, prec + rec)
    acc  = safe_div(tp + tn, tp + tn + fp + fn)
    miou = safe_div(sum_iou, n_iou)
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "Precision": round(prec, 4), "Recall": round(rec, 4),
            "F1-Score": round(f1, 4), "Accuracy": round(acc, 4),
            "Mean IoU": round(miou, 4), "mAP@0.5": round(map50, 4)}


def table_micro(df, gt_all, pred_all, label_col, label):
    """Satu baris: jumlah one-vs-rest seluruh kelas (micro) untuk subset df."""
    tp = tn = fp = fn = 0
    for c in CLASS_NAMES:
        a, b, d, e = ovr_counts(df, c)
        tp += a; tn += b; fp += d; fn += e
    tp_df = df[df["is_tp"]]
    ap = compute_ap50(gt_all, pred_all, list(CLASS_NAMES))
    row = {label_col: label}
    row.update(metric_row(tp, tn, fp, fn, tp_df["IoU"].sum(), len(tp_df), mean_ignore_nan(ap.values())))
    return row


def build_tables(df, gt_all, pred_all):
    class_ids = list(CLASS_NAMES)
    ap_all = compute_ap50(gt_all, pred_all, class_ids)

    # ---------- 1) Per kelas ----------
    rows = []
    for c, name in CLASS_NAMES.items():
        tp, tn, fp, fn = ovr_counts(df, c)
        t = df[(df["gt_cls"] == c) & df["is_tp"]]
        r = {"Skenario/Kelas": f"{SCENARIO_NAME} - {name}"}
        r.update(metric_row(tp, tn, fp, fn, t["IoU"].sum(), len(t), 0.0 if np.isnan(ap_all[c]) else ap_all[c]))
        rows.append(r)
    tot = table_micro(df, gt_all, pred_all, "Skenario/Kelas", f"{SCENARIO_NAME} - TOTAL")
    rows.append(tot)
    df_class = pd.DataFrame(rows)

    # ---------- 2) Per pencahayaan (seluruh kelas) ----------
    rows = []
    for light in LIGHT_ORDER:
        sub = df[df["Lighting"] == light]
        if sub.empty:
            continue
        sub_gt   = [g for g in gt_all if g["light"] == light]
        sub_pred = [p for p in pred_all if p["light"] == light]
        r = table_micro(sub, sub_gt, sub_pred, "Cahaya", light)
        r["Jumlah Gambar"] = len(sub)
        rows.append(r)
    r = table_micro(df, gt_all, pred_all, "Cahaya", "TOTAL")
    r["Jumlah Gambar"] = len(df)
    rows.append(r)
    df_light = pd.DataFrame(rows)
    cols = ["Cahaya", "Jumlah Gambar"] + [c for c in df_light.columns if c not in ("Cahaya", "Jumlah Gambar")]
    df_light = df_light[cols]

    # ---------- 3) Confusion matrix ----------
    col_labels = list(CLASS_NAMES.values()) + [BG_COL]
    row_labels = list(CLASS_NAMES.values())
    if (df["gt_cls"].isna()).any():
        row_labels.append(BG_ROW)
    cm = pd.DataFrame(0, index=row_labels, columns=col_labels)
    for _, r in df.iterrows():
        actual = CLASS_NAMES.get(r["gt_cls"], BG_ROW)
        if r["is_tp"] or (r["pred_cls"] is not None and not pd.isna(r["pred_cls"]) and r["pred_cls"] != r["gt_cls"]):
            pred = CLASS_NAMES[int(r["pred_cls"])]
        else:
            pred = BG_COL    # tidak ada prediksi, atau IoU < 0.5
        cm.loc[actual, pred] += 1
    df_cm = cm.reset_index().rename(columns={"index": "Aktual / Prediksi"})

    return df_class, df_light, df_cm, cm


# =============================================================================
# ANOTASI GAMBAR & HEATMAP
# =============================================================================
def annotate_images(img_out_dir, df):
    colors = {"TP": (0, 255, 0), "TN": (255, 200, 0)}
    for _, r in df.iterrows():
        base = os.path.splitext(r["Image Name"])[0]
        path = next((os.path.join(img_out_dir, base + e) for e in IMG_EXTS
                     if os.path.exists(os.path.join(img_out_dir, base + e))), None)
        if not path:
            continue
        img = cv2.imread(path)
        if img is None:
            continue
        color = colors.get(r["Status"], (0, 0, 255))
        l1 = f"Status: {r['Status']}"
        l2 = f"Aktual: {r['Actual']} | Pred: {r['Prediction']} | IoU: {r['IoU']:.2f}"
        for txt, y, sc, col in [(l1, 35, 0.75, color), (l2, 60, 0.55, (255, 255, 255))]:
            cv2.putText(img, txt, (15, y), cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, txt, (15, y), cv2.FONT_HERSHEY_SIMPLEX, sc, col, 2, cv2.LINE_AA)
        cv2.imwrite(path, img)


def save_confusion_heatmap(cm, model_name, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[INFO] matplotlib tidak ada, heatmap dilewati.")
        return
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.imshow(cm.values, cmap="Blues")
    ax.set_xticks(range(len(cm.columns)), cm.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(cm.index)), cm.index)
    ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual")
    ax.set_title(f"Confusion Matrix - {model_name} ({SCENARIO_NAME})")
    thr = cm.values.max() / 2 if cm.values.max() > 0 else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm.values[i, j]
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > thr else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# =============================================================================
# LOG
# =============================================================================
def write_log(log_path, model_name, df_class, df_light, df_cm, df):
    w = 170
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("=" * w + "\n")
        f.write(f" EVALUATION LOG - {SCENARIO_NAME} - MODEL {model_name}\n")
        f.write(f" Generated at: {datetime.now():%Y-%m-%d %H:%M:%S} | "
                f"IoU>={IOU_THRESHOLD} | Conf>={CONF_THRESHOLD} | Total gambar: {len(df)}\n")
        f.write("=" * w + "\n\n")

        f.write("[ METRIK DETEKSI DAN KLASIFIKASI PER KELAS (ONE-vs-REST) ]\n")
        f.write(df_class.to_string(index=False) + "\n\n")

        f.write("[ RINCIAN PERFORMA BERDASARKAN KONDISI PENCAHAYAAN ]\n")
        f.write(df_light.to_string(index=False) + "\n\n")

        f.write("[ CONFUSION MATRIX EVALUASI DETEKSI (Aktual vs Prediksi) ]\n")
        f.write(df_cm.to_string(index=False) + "\n\n")

        f.write("[ DETAIL PER GAMBAR ]\n")
        f.write(f"{'No':<5} | {'Image Name':<40} | {'Lighting':<8} | {'Aktual':<14} | "
                f"{'Prediksi':<14} | {'IoU':<6} | {'Status'}\n")
        f.write("-" * w + "\n")
        current, idx = None, 0
        for _, r in df.iterrows():
            if r["Lighting"] != current:
                current, idx = r["Lighting"], 0
                f.write(f"\n--- KELOMPOK PENCAHAYAAN: {current.upper()} ---\n")
            idx += 1
            f.write(f"{idx:<5} | {r['Image Name']:<40} | {r['Lighting']:<8} | {r['Actual']:<14} | "
                    f"{r['Prediction']:<14} | {r['IoU']:<6.4f} | {r['Status']}\n")
        f.write("-" * w + "\n\n\n")


# =============================================================================
# MAIN
# =============================================================================
def main():
    open(LOG_OUT, "w", encoding="utf-8").close()
    image_map = build_image_map()

    for model_name, model_path in MODELS.items():
        if not os.path.exists(model_path):
            print(f"[WARNING] Model tidak ditemukan: {model_path}")
            continue

        label_dir, img_out, project_dir = run_inference(model_path, model_name)
        df, gt_all, pred_all = evaluate(label_dir, image_map)
        df_class, df_light, df_cm, cm = build_tables(df, gt_all, pred_all)

        annotate_images(img_out, df)
        save_confusion_heatmap(cm, model_name, os.path.join(project_dir, "confusion_matrix.png"))
        write_log(LOG_OUT, model_name, df_class, df_light, df_cm, df)

        print(f"\n=== {model_name} | {SCENARIO_NAME} ===")
        print(df_class.to_string(index=False))
        print("\n--- PER PENCAHAYAAN ---")
        print(df_light.to_string(index=False))
        print("\n--- CONFUSION MATRIX ---")
        print(df_cm.to_string(index=False))

    print(f"\n[SELESAI] Log tersimpan di: {LOG_OUT}")


if __name__ == "__main__":
    main()
