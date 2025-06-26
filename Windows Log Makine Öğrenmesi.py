import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from minisom import MiniSom
import numpy as np

# Pandas ile veri setini okuyoruz.
veri = pd.read_csv(r"C:\Users\atkn_\Desktop\Bitirme-Son\loghub-master\Windows\Windows_2k_features.csv")

# Özellikleri standardize et çünkü mesafeye dayalı algoritmalar (LOF, DBSCAN, SOFM) için tüm özelliklerin eşit aralıkta olması gerekir.
olceklendirici = StandardScaler()
veri_olcekli = olceklendirici.fit_transform(veri)


# Isolation Forest (Ağaç tabanlı gözetimsiz anomali tespiti)
# 'contamination=auto' → veri içindeki anomali oranı otomatik tahmin edilir.
iforest_modeli = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
etiket_iforest = iforest_modeli.fit_predict(veri_olcekli)
etiket_iforest = (etiket_iforest == -1).astype(int)  # -1 = anomali

print("\n=== Isolation Forest ===")
print(f"Anomali Sayısı: {np.sum(etiket_iforest)} / {len(etiket_iforest)}")


# Local Outlier Factor (K-NN tabanlı anomali tespiti)
# Her noktanın lokal yoğunluğunu komşularıyla karşılaştırır
lof_modeli = LocalOutlierFactor(n_neighbors=20, contamination='auto')
etiket_lof = lof_modeli.fit_predict(veri_olcekli)
etiket_lof = (etiket_lof == -1).astype(int)

print("\n=== Local Outlier Factor ===")
print(f"Anomali Sayısı: {np.sum(etiket_lof)} / {len(etiket_lof)}")


# DBSCAN (Yoğunluk tabanlı kümeleme ile anomali tespiti)
# eps: Komşuluk yarıçapı, min_samples: bir noktayı çekirdek saymak için gereken komşu sayısı
dbscan_modeli = DBSCAN(eps=2.0, min_samples=10)
etiket_dbscan = (dbscan_modeli.fit_predict(veri_olcekli) == -1).astype(int)

print("\n=== DBSCAN ===")
print(f"Anomali Sayısı: {np.sum(etiket_dbscan)} / {len(etiket_dbscan)}")


# SOFM-DIST (Self-Organizing Feature Map ile uzaklığa dayalı tespit)
# MiniSom kullanılarak bir 10x10 harita üzerinde öğrenme yapılır.
som = MiniSom(x=10, y=10, input_len=veri_olcekli.shape[1], sigma=1.0, learning_rate=0.5, random_seed=42)
som.train_random(veri_olcekli, 1000)

# Her nöronun çevresiyle olan ağırlık farkını temsil eden distance map elde edilir.
distance_map = som.distance_map()

# Distance map düzleştirilir ve %90 üzerindeki uzaklıklar anomali kabul edilir.
esik_uzaklik = np.percentile(distance_map.flatten(), 90)

# Her veri noktası için, kazanan (BMU) nöronun distance map değeri üzerinden anomali tespiti yapılır.
etiket_sofm = []
for ornek in veri_olcekli:
    bmu = som.winner(ornek)
    uzaklik = distance_map[bmu]
    etiket_sofm.append(1 if uzaklik > esik_uzaklik else 0)

print("\n=== SOFM-DIST ===")
print(f"Anomali Sayısı: {sum(etiket_sofm)} / {len(etiket_sofm)}")


# Sonuçları birleştir ve CSV dosyasına kaydet
veri_sonuclari = veri.copy()
veri_sonuclari['IForest'] = etiket_iforest
veri_sonuclari['LOF'] = etiket_lof
veri_sonuclari['DBSCAN'] = etiket_dbscan
veri_sonuclari['SOFM_DIST'] = etiket_sofm

# Sonuç dosyası olarak kaydedilir
cikis_dosyasi = r"C:\Users\atkn_\Desktop\Bitirme-Son\loghub-master\Windows\Windows_2k_anomaly_results.csv"
veri_sonuclari.to_csv(cikis_dosyasi, index=False)

print(f"Sonuçlar başarıyla kaydedildi: {cikis_dosyasi}")