import pandas as pd
import re

# Yapılandırılmış HDFS log dosyasını pandas ile okuyoruz.
# Bu dosya, her log satırını EventId ve Content gibi sütunlara ayırır
log_dosyasi = pd.read_csv("C:/Users/atkn_/Desktop/AAA/yapilandirilmis/HDFS/HDFS_2k.log_structured.csv")

# 'Content' sütunundan BlockId değerini düzenli ifadeyle çıkartıyoruz.
# BlockId formatı: blk_-1234567890 gibi olabilir (negatif veya pozitif tam sayı)
# Eğer içerikte BlockId yoksa 'none' yazıyoruz.
log_dosyasi['BlokID'] = log_dosyasi['Content'].apply(
    lambda satir: re.search(r'blk_-?\d+', str(satir)).group() if re.search(r'blk_-?\d+', str(satir)) else 'none'
)

# BlockId bazlı etiket (anomaly/normal) dosyasını pandas ile okuyoruz
# Bu dosya, her blok için 0 (normal) veya 1 (anomaly) etiketini içerir
etiket_dosyasi = pd.read_csv("C:/Users/atkn_/Desktop/AAA/loghub/HDFS/preprocessed/anomaly_label.csv")

# Log dosyası ile etiketleri BlockId üzerinden birleştiriyoruz. (INNER JOIN)
# Sadece etiketlenmiş bloklar kalır; etiketlenmemiş olanları atıyoruz.
birlesik_veri = log_dosyasi.merge(etiket_dosyasi, on="BlockId", how="inner")

# Her BlockId için EventId frekanslarına göre özellik matrisi oluştur
# Örnek: BlockID = blk_123 için hangi EventId kaç kez geçmiş? → satır: BlockID, sütun: EventId sayısı
ozellik_matrisi = birlesik_veri.groupby('BlockId')['EventId']\
    .value_counts().unstack().fillna(0)

# Her BlockId'ye ait etiketi de özellik matrisine ekle (tüm satırlar aynı etiket olduğundan max alınabilir)
ozellik_matrisi['Etiket'] = birlesik_veri.groupby('BlockId')['Label'].max()

# Sonuçları CSV olarak dışa aktarıyoruz.
cikis_yolu = "C:/Users/atkn_/Desktop/AAA/features_hdfs_labeled.csv"
ozellik_matrisi.to_csv(cikis_yolu)

# İşlem tamamlandıktan sonra bilgilendirme mesajı geçiyoruz
print("Gerçek etiketli özellik vektörü dosyası başarıyla oluşturuldu.")
