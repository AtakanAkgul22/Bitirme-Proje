from minisom import MiniSom
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Pandas ile veriyi okuyoruz.
veri = pd.read_csv("C:/Users/atkn_/Desktop/AAA/features_hdfs_labeled.csv")

#  Özellik (X) ve etiket (y) ayırma işlemi.
# 'BlockId' ve 'label' dışında kalan sütunlar özellik vektörünü oluşturur.
# Etiketler: Normal → 0, Anomaly → 1 olarak sayısallaştırılır.
ozellikler = veri.drop(columns=["label", "BlockId"], errors="ignore")
gercek_etiketler = veri["label"].map({"Normal": 0, "Anomaly": 1})

# Özellikleri normalize ediyoruz. (standardizasyon)
# Bu işlem, SOFM gibi mesafe temelli algoritmalarda tüm değişkenlerin eşit ölçeğe getirilmesini sağlar.
olceklendirici = StandardScaler()
ozellikler_olcekli = olceklendirici.fit_transform(ozellikler)

# SOFM (Self-Organizing Feature Map) modeli oluşturuyoruz.
# MiniSom, örüntülerin iki boyutlu haritada gruplandırılmasını sağlar.
# x=10, y=10: Harita boyutu (100 nöron), input_len: Giriş vektörünün boyutu.
som = MiniSom(
    x=10,
    y=10,
    input_len=ozellikler_olcekli.shape[1],
    sigma=1.0,
    learning_rate=0.5,
    random_seed=42
)
som.random_weights_init(ozellikler_olcekli)  # Başlangıç ağırlıkları rastgele atanır
som.train_random(ozellikler_olcekli, 1000)   # 1000 iterasyonluk rastgele eğitim yapılır

# SOFM çıktılarını elde ediyoruz.
# Her giriş vektörü için en yakın nöron (BMU - Best Matching Unit) koordinatı alınır.
# Örneğin (4, 7) gibi bir 2D grid koordinatı.
bmu_koordinatlari = np.array([som.winner(x) for x in ozellikler_olcekli])

# Koordinatları normalize et: Harita boyutu 10 olduğundan (0,10) aralığını (0,1)'e çevir.
# Bu normalizasyon DBSCAN’in eps parametresiyle daha stabil çalışmasını sağlar.
bmu_koordinatlari_normalize = bmu_koordinatlari / 10.0

# DBSCAN kümeleme algoritmasını uyguluyoruz.
# DBSCAN, yoğunluk tabanlı bir kümeleme algoritmasıdır ve -1 etiketini outlier (anomaly) olarak verir.
# eps: Komşuluk yarıçapı, min_samples: Bir noktayı çekirdek nokta saymak için minimum komşu sayısı.
dbscan_modeli = DBSCAN(eps=0.2, min_samples=5)
dbscan_etiketleri = dbscan_modeli.fit_predict(bmu_koordinatlari_normalize)

# DBSCAN çıktısında -1 olarak işaretlenen örnekler anomali kabul edilir
# Bu nedenle tahminler 0 (normal) ve 1 (anomaly) olarak dönüştürülür
sofm_dbscan_tahmin = (dbscan_etiketleri == -1).astype(int)

# Performans değerlendirmesi kısmı.
# Gerçek etiketlerle modelin tahmin ettiği etiketler karşılaştırılır.
# Precision, recall, f1-score gibi metrikler yazdırılır.
print("\n🔹 SOFM-DBSCAN Sonuçları")
print(classification_report(
    gercek_etiketler,
    sofm_dbscan_tahmin,
    target_names=["Normal", "Anomali"],
    zero_division=0
))

