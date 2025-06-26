import pandas as pd  

# Dosya yollarını tanımlıyoruz.
# Giriş dosyası: Daha önce EventId bazında gruplanmış ve etiketlenmiş log verisi
# Çıkış dosyası: 1000 satırlık örnek (sample) veri seti
birlesik_veri_yolu = r"C:\Users\atkn_\Desktop\Bitirme-Son\loghub-master\HDFS\preprocessed\HDFS_10k_merged.csv"
ornek_veri_cikis_yolu = r"C:\Users\atkn_\Desktop\Bitirme-Son\loghub-master\HDFS\preprocessed\HDFS_1k_sample.csv"

# pandas kullanarak veriyi okuyoruz.
veri = pd.read_csv(birlesik_veri_yolu)

# Anomali (Label=1) ve normal (Label=0) örnekleri ayırıyoruz.
anomaliler = veri[veri['Label'] == 1]  # Gerçek anomali içeren bloklar
normaller = veri[veri['Label'] == 0]   # Normal olan bloklar

# Toplamda oluşturulacak örneklem veri seti büyüklüğü (1000 satır)
# Anomali sayısı veride ne kadarsa tamamı alınacak
# Geriye kalan satırlar normal sınıftan rastgele seçilecek
toplam_ornek_sayisi = 1000
anomal_sayisi = len(anomaliler)               # Örneğin: 413 satır
normal_ornek_sayisi = toplam_ornek_sayisi - anomal_sayisi  # Geriye kalan: 587 satır

# Normal sınıftan rastgele örnek seçimi. (karışıklığı önlemek için random_state sabit tutulur)
rastgele_normaller = normaller.sample(n=normal_ornek_sayisi, random_state=42)

# Anomaliler ile seçilen normal örnekler birleştirilir ve satırlar karıştırılır.
orneklenmis_veri = pd.concat([anomaliler, rastgele_normaller])\
                     .sample(frac=1, random_state=42)\
                     .reset_index(drop=True)

# Yeni örneklem veri seti CSV olarak kaydediyoruz.
orneklenmis_veri.to_csv(ornek_veri_cikis_yolu, index=False)

# Bilgilendirme mesajı yazdırılır.
print(f"Yeni 1000 satırlık örnek veri seti başarıyla oluşturuldu ve kaydedildi: {ornek_veri_cikis_yolu}")
