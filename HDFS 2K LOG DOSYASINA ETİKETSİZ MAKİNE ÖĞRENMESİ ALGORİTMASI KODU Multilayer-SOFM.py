from minisom import MiniSom
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Pandas ile veriyi okuyoruz.
veri = pd.read_csv("C:/Users/atkn_/Desktop/AAA/features_hdfs_labeled.csv")

# Özellikleri ve etiketleri ayırıyoruz.
# Etiket: Normal → 0, Anomaly → 1 olacak şekilde sayısal forma çeviriyoruz.
ozellikler = veri.drop(columns=["label", "BlockId"], errors="ignore")
gercek_etiketler = veri["label"].map({"Normal": 0, "Anomaly": 1})

# Özellikleri standartlaştırıyoruz çünkü SOFM (Self-Organizing Feature Map) mesafe temelli çalıştığı için tüm özelliklerin eşit katkı sağlaması gerekir
olceklendirici = StandardScaler()
ozellikler_olcekli = olceklendirici.fit_transform(ozellikler)

# İlk katman SOFM modelini tanımlıyoruz. (Giriş: Ham özellik vektörü → Çıkış: 2D BMU koordinatları)
# 10x10’luk bir harita üzerinde MiniSom ağı eğitiliyor
# input_len = özellik vektörünün boyutu, yani her örnekte kaç adet EventId sayısı varsa o kadar
som1 = MiniSom(x=10, y=10, input_len=ozellikler_olcekli.shape[1], sigma=1.0, learning_rate=0.5, random_seed=42)
som1.random_weights_init(ozellikler_olcekli)  # Ağırlıklar rastgele başlatılır
som1.train_random(ozellikler_olcekli, 1000)   # Eğitim süreci başlatılır

# İlk SOFM’in çıktısı: Her örneğin haritadaki BMU koordinatlarını alıyoruz.
# Örneğin [5,3] gibi bir koordinata en yakın olduğunu belirtiyor
# Daha sonra bu koordinatlar normalize edilerek ikinci katmana veriliyor.
bmu1_koordinatlari = np.array([som1.winner(x) for x in ozellikler_olcekli])   # shape: (örnek sayısı, 2)
bmu1_normalize = bmu1_koordinatlari / 10.0  # 10x10 harita olduğundan 0–1 aralığına normalize ediliyor

# İkinci katman SOFM modelini tanımlıyoruz. (İlk katmanın BMU koordinatları)
# Bu katman, örüntülerin harita üzerindeki konumsal yapısını özetlemek üzere daha küçük bir grid (5x5) kullanır.
som2 = MiniSom(x=5, y=5, input_len=2, sigma=0.5, learning_rate=0.3, random_seed=42)
som2.random_weights_init(bmu1_normalize)
som2.train_random(bmu1_normalize, 500)

# Her örnek için ikinci katmandaki en iyi eşleşen nöronu (BMU) alıyoruz.
# Ardından bu BMU’ların kaç kez aktif olduğuna dair sayım yapılır.
bmu2_koordinatlari = [tuple(som2.winner(x)) for x in bmu1_normalize]  # her biri (x,y) koordinatı
bmu2_sayim = Counter(bmu2_koordinatlari)  # her nöronun kaç kez aktif olduğu
ornek_bmu2_aktivasyonlari = np.array([bmu2_sayim[bmu] for bmu in bmu2_koordinatlari])

# Aktivasyon eşik değeri ile anomali tespiti kısmı.(SOFM-COUNT prensibi)
# En az aktif olan nöronlara düşen örnekler potansiyel anomalidir.
# İlk %5’e düşen aktivasyonlara sahip örnekler anomali sayılır.
esik = np.percentile(ornek_bmu2_aktivasyonlari, 5)
tahmin_edilen_etiketler = (ornek_bmu2_aktivasyonlari <= esik).astype(int)

# Model çıktısını değerlendirme.
# Precision, recall, f1-score gibi metriklerle performans analizi yapılır.
print("\n🔹 Multilayer-SOFM Sonuçları")
print(classification_report(
    gercek_etiketler,
    tahmin_edilen_etiketler,
    target_names=["Normal", "Anomali"],
    zero_division=0
))
