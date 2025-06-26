import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

#Özellik vektörleri içeren CSV dosyasını pandas ile okuyoruz.
veri = pd.read_csv("C:/Users/atkn_/Desktop/AAA/features_hdfs_labeled.csv")

# Etiket (label) ve BlockId sütunlarını çıkararak sadece özellikleri al
# 'label' = anomali olup olmadığını belirten etiket
# 'BlockId' = her log grubuna ait benzersiz kimlik
ozellikler = veri.drop(columns=["label", "BlockId"], errors="ignore")
gercek_etiketler = veri["label"]

# Etiketleri sayısal değerlere dönüştürüyoruz. (Normal:0, Anomaly:1)
gercek_etiketler = gercek_etiketler.map({"Normal": 0, "Anomaly": 1})

# Özellikleri normalize ediyoruz. (ölçeklendirme) çünkü KMeans ve DBSCAN gibi mesafe tabanlı algoritmalar için gereklidir.
olceklendirici = StandardScaler()
ozellikler_olcekli = olceklendirici.fit_transform(ozellikler)

# K-Means kümeleme algoritmasını uyguluyoruz.
# n_clusters=2 → biri normal, diğeri anomali kümesi olarak varsayılır.
kume_modeli = KMeans(n_clusters=2, random_state=42)
kume_etiketleri = kume_modeli.fit_predict(ozellikler_olcekli)

# Kümeleri yorumluyoruz: Küçük olan küme anomali kümesi olarak işaretlenir.
# Bu yöntem tamamen tahmine dayalıdır; etiket bilgisi kullanılmaz.
if np.sum(kume_etiketleri == 0) < np.sum(kume_etiketleri == 1):
    kmeans_tahmin = (kume_etiketleri == 0).astype(int)
else:
    kmeans_tahmin = (kume_etiketleri == 1).astype(int)

# DBSCAN algoritmasını uyguluyoruz.
# DBSCAN, gürültülü noktaları (outlier) -1 olarak etiketler; bunlar anomalidir
dbscan_modeli = DBSCAN(eps=1.5, min_samples=5)
dbscan_etiketleri = dbscan_modeli.fit_predict(ozellikler_olcekli)
dbscan_tahmin = (dbscan_etiketleri == -1).astype(int)

# K-Means değerlendirme metriklerini yazdırıyoruz.
print("\n🔹 K-Means Sonuçları")
print(classification_report(
    gercek_etiketler,
    kmeans_tahmin,
    target_names=["Normal", "Anomali"],
    zero_division=0
))

# DBSCAN değerlendirme metriklerini yazdırıyoruz.
print("\n🔹 DBSCAN Sonuçları")
print(classification_report(
    gercek_etiketler,
    dbscan_tahmin,
    target_names=["Normal", "Anomali"],
    zero_division=0
))
