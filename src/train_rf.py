from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def find_best_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[float, float]:
    """
    Amaç: 0/1 sınıflaması için en iyi eşik (threshold) değerini bulmak.

    Neyi neden yapıyoruz?
    - RandomForest 'predict_proba' ile 0-1 arası olasılık üretir.
    - Varsayılan eşik 0.50'dir ama her veri seti için ideal olmayabilir.
    - Biz burada farklı eşikleri deneyip F1 skorunu maksimize eden eşiği seçiyoruz.

    Dönüş:
    - best_thr: F1'i en yüksek yapan eşik
    - best_f1: o eşiğe karşılık gelen F1 skoru
    """
    thresholds = np.linspace(0.05, 0.95, 91)  # 0.05 - 0.95 arası tarama
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

    # 1) Dosya yolu (neden pathlib?)
    # - Script'i nereden çalıştırırsan çalıştır, path bozulmasın diye.
    BASE_DIR = Path(__file__).resolve().parent.parent  # src/.. => proje root
    file_path = BASE_DIR / "data" / "raw" / "pah_balanced_58x58_anonymized.csv"

    # 2) Veri yükle
    df = pd.read_csv(file_path)

    # 3) Target kontrolü
    if "target" not in df.columns:
        raise ValueError("CSV içinde 'target' kolonu bulunamadı. Kolon adını kontrol et.")

    # 4) X / y ayır
    # Neyi neden yapıyoruz?
    # - 'target' dışındaki tüm kolonlar feature (X)
    # - target: 0 benign, 1 patojenik (y)
    X = df.drop(columns=["target"])
    y = df["target"].astype(int)

    # 5) Train / Test split (stratify neden?)
    # - Sınıf oranı korunur (benign/patojenik dengesi bozulmasın)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # 6) Model
    # Neyi neden yapıyoruz?
    # - n_estimators: ağaç sayısı yüksek olsun => daha stabil
    # - class_weight: dengesizlik olursa otomatik dengeye yardımcı
    # - n_jobs=-1: tüm çekirdekleri kullan => hız
    rf = RandomForestClassifier(
        n_estimators=600,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    rf.fit(X_train, y_train)

    # 7) Olasılık tahmini
    y_proba = rf.predict_proba(X_test)[:, 1]

    # 8) Optimal threshold bul
    best_thr, best_f1 = find_best_threshold(y_test.to_numpy(), y_proba)

    # 9) Threshold ile final sınıf tahmini
    y_pred = (y_proba >= best_thr).astype(int)

    # 10) Metrikler
    acc = accuracy_score(y_test, y_pred) * 100.0
    roc_auc = roc_auc_score(y_test, y_proba)

    # 11) Terminal çıktısı (CatBoost ekranı gibi)
    print("==============================================")
    print(f"ENSEMBLE FINAL BAŞARISI: %{acc:.2f}")
    print(f"OPTIMAL EŞİK DEĞERİ: {best_thr:.2f}")
    print(f"BEST F1 (THRESHOLD TARAMA): {best_f1:.2f}")
    print(f"ROC-AUC: {roc_auc:.3f}")
    print("==============================================\n")

    print("Final Sınıflandırma Raporu:\n")
    print(classification_report(y_test, y_pred, digits=2))


if __name__ == "__main__":
    main()