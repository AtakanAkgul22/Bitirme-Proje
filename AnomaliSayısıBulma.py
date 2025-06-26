import pandas as pd 

# Birleştirilmiş HDFS log verisinin dosya yolu tanımlanır.
# Bu dosya, EventId'lerin belirli bir BlockId altında gruplanmış hâlini ve 'Label' sütunundaki etiket bilgilerini içerir
birlesik_dosya_yolu = r"C:\Users\atkn_\Desktop\Bitirme-Son\loghub-master\HDFS\preprocessed\HDFS_2k_merged.csv"

# pandas ile veriyi okuyoruz.
veri = pd.read_csv(birlesik_dosya_yolu)

# İlk 5 satır ekrana yazdırılır.
# Amaç: Verinin yapısını, sütun adlarını ve örnek kayıtları hızlıca incelemek
print(veri.head())

# 'Label' sütunundaki etiketlerin (Normal/Anomaly) kaç kez geçtiği sayılır ve yazdırılır
# Bu, sınıf dağılımını görmek için önemlidir. Dengesiz dağılım varsa model eğitimi etkilenebilir.
print(veri['Label'].value_counts())
