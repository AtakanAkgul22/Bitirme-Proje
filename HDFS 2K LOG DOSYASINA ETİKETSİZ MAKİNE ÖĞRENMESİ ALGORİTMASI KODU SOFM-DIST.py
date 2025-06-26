from minisom import MiniSom
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Pandas ile veriyi okuyoruz.
veri = pd.read_csv("C:/Users/atkn_/Desktop/AAA/features_hdfs_labeled.csv")

# Özellikleri ve etiketleri ayırıyoruz.
# 'label' ve 'BlockId' sütunları analiz için gerekli değildir. Model sadece sayısal özellik vektörleri ile çalışır.
# Bu nedenle sadece EventID frekanslarının bulunduğu sütunlar alıyoruz..
ozellikler = veri.drop(columns=["label", "BlockId"], errors="ignore")

# Etiket bilgisi (Normal, Anomaly) sayısal forma çevririyoruz.
# Bu işlem değerlendirme metriklerinde kullanılacak olan gerçek etiket kümesini oluşturur.
gercek_etiketler = veri["label"].map({"Normal": 0, "Anomaly": 1})

# Özellikleri standardize ediyoruz. Çünkü SOFM gibi mesafe tabanlı algoritmalarda her özelliğin aynı skala aralığında olması önemlidir.
# Aksi takdirde büyük değere sahip değişkenler mesafeye daha fazla etki eder.
olceklendirici = StandardScaler()
ozellikler_olcekli = olceklendirici.fit_transform(ozellikler)

# MiniSom (SOFM) modelini tanımlıyoruz.
# x=10, y=10: SOM haritasının boyutlarıdır. Toplamda 100 nöronluk bir grid.
# input_len: Giriş vektörünün boyutu, yani kaç adet EventID özelliği varsa o kadar.
# sigma: Komşuluk yarıçapı (nöronlara yayılma etkisi); yüksek sigma daha yaygın bir etki yaratır.
# learning_rate: Öğrenme oranı; büyük değerler hızlı öğrenmeye, küçük değerler daha hassas güncellemeye neden olur.
# random_seed: Rastgeleliğe bağlı davranışı tekrarlanabilir hale getirmek için sabit tutulur.
som = MiniSom(
    x=10, y=10,
    input_len=ozellikler_olcekli.shape[1],
    sigma=1.0,
    learning_rate=0.5,
    random_seed=42
)

# SOFM ağındaki her nöronun ağırlıklarını rastgele başlatıyoruz.
som.random_weights_init(ozellikler_olcekli)

# SOFM eğitim süreci başlatıyoruz.
# 1000 iterasyon boyunca her seferinde rastgele bir örnek seçilerek ağ eğitilecek.
# Bu süreçte hem en yakın nöronun ağırlıkları hem de komşularının ağırlıkları güncellenecek.
som.train_random(ozellikler_olcekli, 1000)

# Her veri örneği için En İyi Eşleşen Nöron'a (BMU) olan öklidyen mesafeyi hesaplanacak.
# Her veri noktası için, kendisine en yakın olan nöronun (BMU) ağırlık vektörü bulunacak.
# Bu BMU'nun ağırlığı ile veri örneği arasındaki mesafe hesaplanır.
# Bu mesafe ne kadar büyükse, o örnek SOM haritasına o kadar "yabancıdır" yani potansiyel anomali olabilir.
bmu_uzakliklari = np.array([
    np.linalg.norm(
        veri_noktasi - som.get_weights()[som.winner(veri_noktasi)[0]][som.winner(veri_noktasi)[1]]
    )
    for veri_noktasi in ozellikler_olcekli
])

# Anomali eşik değeri (threshold) belirlenecek.
# Uzaklıkların %95'inden büyük olanlar anomali kabul edilecek çünkü normal örnekler SOM haritasındaki bir BMU’ya yakın olurken, anomaliler genellikle daha uzak konumlanır.
# Bu yüzden en uzak %5’lik dilim anomali olarak tanımlanır.
esik_uzaklik = np.percentile(bmu_uzakliklari, 95)

# Eşik değeriyle karşılaştırma.
# Uzaklığı eşik değerinden büyük olan her örnek anomali (1), değilse normal (0) olarak etiketlenecek.
som_tahminleri = (bmu_uzakliklari >= esik_uzaklik).astype(int)

# Sınıflandırma başarı metriklerini yazdırılacak.
# precision, recall, f1-score ve destek (support) gibi metrikler gösterilir
# zero_division=0: Eğer bir sınıf için tahmin yoksa hata yerine sıfırla doldurulacak.
print("\n🔹 SOFM-DIST Sonuçları")
print(classification_report(
    gercek_etiketler,
    som_tahminleri,
    target_names=["Normal", "Anomali"],
    zero_division=0
))

