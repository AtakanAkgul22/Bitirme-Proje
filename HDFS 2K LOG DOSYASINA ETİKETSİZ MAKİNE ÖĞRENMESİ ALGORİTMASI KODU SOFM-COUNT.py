from minisom import MiniSom
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Pandas kullanarak verimizi okuyoruz.
veri = pd.read_csv("C:/Users/atkn_/Desktop/AAA/features_hdfs_labeled.csv")

# Özellik (X) ve etiket (y) ayırma işlemi.
ozellikler = veri.drop(columns=["label", "BlockId"], errors="ignore")
gercek_etiketler = veri["label"].map({"Normal": 0, "Anomaly": 1})

# Özellikleri normalize ediyoruz. Çünkü SOM gibi mesafe tabanlı modellerde ölçeklendirme gereklidir

olceklendirici = StandardScaler()
ozellikler_olcekli = olceklendirici.fit_transform(ozellikler)

# SOFM (Self-Organizing Feature Map) modelini tanımlıyoruz ve eğitiyoruz.
# x, y: SOM harita boyutu (10x10)
# input_len: giriş boyutu (özellik sayısı)
# sigma: komşuluk fonksiyonu genişliği
# learning_rate: öğrenme oranı
# 1000 iterasyon boyunca rastgele örneklerle eğitilir
som = MiniSom(
    x=10, y=10,
    input_len=ozellikler_olcekli.shape[1],
    sigma=1.0,
    learning_rate=0.5,
    random_seed=42
)
som.random_weights_init(ozellikler_olcekli)
som.train_random(ozellikler_olcekli, 1000)

# Her örnek için en iyi eşleşen nöronun (BMU) koordinatlarını alıyoruz.
# Her veri noktasının harita üzerinde en yakın olduğu hücre (BMU) belirlenir.
bmu_koordinatlari = [tuple(som.winner(ornek)) for ornek in ozellikler_olcekli]

# Harita üzerindeki her BMU’nun kaç defa eşleştiğini sayıyoruz.
# Bu sayımlar nöron aktivasyonlarını temsil eder.
bmu_sayilari = Counter(bmu_koordinatlari)

# Her örneğin bağlı olduğu BMU’nun kaç defa aktif olduğunu belirliyoruz.
ornek_bmu_aktivasyonlari = np.array([bmu_sayilari[bmu] for bmu in bmu_koordinatlari])

# Aktivasyon eşiği belirliyoruz.
# En az kullanılan (az aktif) nöronlara düşen örnekler anomali kabul edilir.
# İlk %5’e düşen aktivasyonlara sahip örnekler anomali sayılır.
esik = np.percentile(ornek_bmu_aktivasyonlari, 5)

# Eşik altındakiler anomali olarak etiketlenir (1), diğerleri normaldir (0)
som_tahminleri = (ornek_bmu_aktivasyonlari <= esik).astype(int)

# 8. Sınıflandırma başarı metriklerini yazdırıyoruz.
print("\n🔹 SOFM-COUNT Sonuçları")
print(classification_report(
    gercek_etiketler,
    som_tahminleri,
    target_names=["Normal", "Anomali"],
    zero_division=0
))

