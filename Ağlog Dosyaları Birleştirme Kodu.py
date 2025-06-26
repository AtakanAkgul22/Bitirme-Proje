import os
import pandas as pd
import numpy as np

# Ana dizin tanımlanır: CSV dosyalarının bulunduğu klasör
klasor_yolu = r"C:\Users\atkn_\Desktop\MachineLearningCVE"

# Klasördeki tüm .csv uzantılı dosyalar listelenir.
csv_dosyalari = [dosya for dosya in os.listdir(klasor_yolu) if dosya.endswith(".csv")]

# Her dosya için etiket (Label) dağılımı ve toplam satır sayısı yazdırılır.
for dosya_adi in csv_dosyalari:
    tam_yol = os.path.join(klasor_yolu, dosya_adi)
    veri = pd.read_csv(tam_yol)
    
    print(f"{dosya_adi} - Toplam Satır: {len(veri)}")

    # Etiket sütunu "Label" ya da " Label" olarak farklı yazılmış olabilir, kontrol edilir
    if " Label" in veri.columns or "Label" in veri.columns:
        etiket_sutun = " Label" if " Label" in veri.columns else "Label"
        print("Etiket Dağılımı:")
        print(veri[etiket_sutun].value_counts())
    
    print("-" * 40)

# Tüm CSV dosyaları tek bir DataFrame'e birleştirilir (concat)
birlesmis_veriler = []
for dosya_adi in csv_dosyalari:
    tam_yol = os.path.join(klasor_yolu, dosya_adi)
    veri = pd.read_csv(tam_yol)
    birlesmis_veriler.append(veri)

# Satırları sıfırlayıp, tek bir DataFrame olarak birleştirir.
tum_veri = pd.concat(birlesmis_veriler, ignore_index=True)

# Sütun adlarının başında veya sonunda boşluk varsa temizlenir.
# Bu, özellikle ' Label' gibi sütunları 'Label' haline getirir.
tum_veri.columns = [sutun.strip() for sutun in tum_veri.columns]

# Sonsuz (inf/-inf) değerleri NaN'e çevir ve tüm eksik değerleri kaldır
tum_veri.replace([np.inf, -np.inf], np.nan, inplace=True)
tum_veri = tum_veri.dropna()

# Temizlenmiş veri seti belirtilen konuma CSV olarak kaydedilir.
cikis_dosyasi_yolu = os.path.join(klasor_yolu, "Birlestirilmisdosya.csv")
tum_veri.to_csv(cikis_dosyasi_yolu, index=False)

print(f"Birleştirilmiş ve temizlenmiş CSV başarıyla kaydedildi:\n{cikis_dosyasi_yolu}")
