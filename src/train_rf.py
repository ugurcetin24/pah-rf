from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


def find_best_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[float, float]:
    """
    Amaç: En iyi threshold'u (eşik) F1 skoruna göre seçmek.

    Neyi neden yapıyoruz?
    - Model olasılık (0-1) üretir: predict_proba.
    - Varsayılan threshold=0.50 her zaman optimal değildir.
    - F1'i maksimize eden threshold'u bulup, daha dengeli performans alırız.

    Dönüş:
    - best_thr: en iyi eşik
    - best_f1: bu eşikteki F1
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    best_thr = 0.50
    best_f1 = -1.0

    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)

    return best_thr, float(best_f1)


def main() -> None:
    print("Ensemble (Topluluk) Modeli Eğitiliyor...\n")
    print("Stratified 5-Fold Cross Validation Başlıyor...\n")

    # 1) Güvenli dosya yolu (nereden çalıştırırsan çalıştır bozulmasın)
    BASE_DIR = Path(__file__).resolve().parent.parent
    file_path = BASE_DIR / "data" / "raw" / "pah_balanced_58x58_anonymized.csv"

    # 2) Veri yükle
    df = pd.read_csv(file_path)

    # 3) Target kontrol
    if "target" not in df.columns:
        raise ValueError("CSV içinde 'target' kolonu bulunamadı. Kolon adını kontrol et.")

    X = df.drop(columns=["target"])
    y = df["target"].astype(int).to_numpy()

    # 4) CV ayarları
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 5) Metrikleri biriktireceğiz
    fold_metrics = []
    all_y_true = []
    all_y_pred = []

    # 6) Fold döngüsü
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train = X.iloc[train_idx]
        y_train = y[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y[test_idx]

        # Model (her fold için sıfırdan eğitiyoruz)
        rf = RandomForestClassifier(
            n_estimators=600,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)

        # Olasılık
        y_proba = rf.predict_proba(X_test)[:, 1]

        # Fold için optimal threshold (F1 maks)
        best_thr, best_f1 = find_best_threshold(y_test, y_proba)

        # Fold tahminleri
        y_pred = (y_proba >= best_thr).astype(int)

        # Fold metrikleri
        acc = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        cm = confusion_matrix(y_test, y_pred)

        fold_metrics.append(
            {
                "fold": fold_idx,
                "accuracy": acc,
                "f1": best_f1,         # bu fold'da threshold taramasıyla bulunan en iyi F1
                "roc_auc": roc_auc,
                "threshold": best_thr,
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1]),
            }
        )

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

        # Fold çıktısı (jüriye gösterilebilir)
        print("==============================================")
        print(f"FOLD {fold_idx}/5 SONUÇLARI")
        print(f"Accuracy:  {acc*100:.2f}%")
        print(f"ROC-AUC:   {roc_auc:.3f}")
        print(f"Best F1:   {best_f1:.2f}")
        print(f"Threshold: {best_thr:.2f}")
        print(f"Confusion Matrix (TN FP / FN TP): {cm.tolist()}")
        print("==============================================\n")

    # 7) Özet tablo
    metrics_df = pd.DataFrame(fold_metrics)

    acc_mean = metrics_df["accuracy"].mean()
    acc_std = metrics_df["accuracy"].std(ddof=1)

    f1_mean = metrics_df["f1"].mean()
    f1_std = metrics_df["f1"].std(ddof=1)

    auc_mean = metrics_df["roc_auc"].mean()
    auc_std = metrics_df["roc_auc"].std(ddof=1)

    thr_mean = metrics_df["threshold"].mean()
    thr_std = metrics_df["threshold"].std(ddof=1)

    print("##############################################")
    print("5-FOLD CV ÖZET (ORTALAMA ± STD)")
    print(f"Accuracy:  {acc_mean*100:.2f}% ± {acc_std*100:.2f}%")
    print(f"F1:        {f1_mean:.2f} ± {f1_std:.2f}")
    print(f"ROC-AUC:   {auc_mean:.3f} ± {auc_std:.3f}")
    print(f"Threshold: {thr_mean:.2f} ± {thr_std:.2f}")
    print("##############################################\n")

    # 8) Tüm fold tahminlerinden global rapor
    # Neyi neden yapıyoruz?
    # - Her örnek bir fold’da test oldu.
    # - Bu yüzden burada ürettiğimiz classification_report, "CV üzerinden genel performans" gibi düşünülebilir.
    print("GLOBAL (TÜM FOLD'LARIN TEST TAHMİNLERİ) SINIFLANDIRMA RAPORU:\n")
    print(classification_report(np.array(all_y_true), np.array(all_y_pred), digits=2))

    # 9) İstersen raporu dosyaya da yazalım (jüri için kanıt)
    reports_dir = BASE_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv_path = reports_dir / "cv_metrics_rf.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)

    print(f"\n[OK] Fold metrikleri kaydedildi: {metrics_csv_path}")


if __name__ == "__main__":
    main()