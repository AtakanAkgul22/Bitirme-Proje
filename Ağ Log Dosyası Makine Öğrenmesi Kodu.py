import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Makine öğrenmesi modelleri
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Veri setini okuyoruz ve sütun adlarındaki boşlukları kaldırıyoruz.
veri = pd.read_csv(r"C:\Users\atkn_\Desktop\MachineLearningCVE\ALL_CICIDS2017_CLEAN.csv")
veri.columns = [sutun.strip() for sutun in veri.columns]  # Bazı sütun adlarında boşluk olabilir (ör: ' Label')

# Etiket sütunundaki boşluk karakterlerini kaldırıyoruz. (örneğin: 'DoS Hulk ' → 'DoS Hulk')
veri['Label'] = veri['Label'].str.strip()

# Sadece sayısal sütunları alıyoruz çünkü kullanacağımız modeller sadece sayısal verilerle çalışıyor.
sayisal_sutunlar = veri.select_dtypes(include=[float, int]).columns
X = veri[sayisal_sutunlar]   # Özellik vektörü (bağımsız değişkenler)
y = veri['Label']            # Hedef (bağımlı değişken)

# Etiketleri sayısal forma çeviriyoruz. (model girişleri için gerekli)
etiket_donusturucu = LabelEncoder()
y_sayisal = etiket_donusturucu.fit_transform(y)  # Örn: 'DoS Hulk' → 3

# Veriyi eğitim ve test setine ayırıyoruz.
X_egitim, X_test, y_egitim_sayisal, y_test_sayisal = train_test_split(
    X, y_sayisal,
    test_size=0.2,          # Verinin %20'si test için ayrılır
    stratify=y_sayisal,     # Sınıf dağılımını eğitim/test setinde korur
    random_state=42         # Aynı veri bölünmesinin tekrar elde edilmesini sağlar
)

# Test ve eğitim etiketlerini orijinal (string) forma geri döndürüyoruz.
y_egitim = etiket_donusturucu.inverse_transform(y_egitim_sayisal)
y_test = etiket_donusturucu.inverse_transform(y_test_sayisal)

# Kullanılacak modeller ve açıklamalı parametreleri
modeller = [
    (
        "Rasgele Orman",
        RandomForestClassifier(
            n_estimators=100,         # Oluşturulacak ağaç sayısı → genelde 100 iyi bir başlangıç noktasıdır.
            random_state=42,          # Rastgeleliğin kontrolü için sabit değer.
            class_weight='balanced'   # Sınıf dengesizliğini otomatik olarak dengeler. (özellikle azınlık sınıflar için önemlidir)
        )
    ),
    (
        "XGBoost",
        XGBClassifier(
            n_estimators=100,         # 100 boosting round (karar ağacı)
            random_state=42,          # Tekrarlanabilir sonuçlar için
            tree_method='hist',       # Histogram temelli yapı → daha hızlı ve daha az bellek kullanımı sağlar
            verbosity=0               # Konsola uyarı ya da bilgi basmaz
        )
    ),
    (
        "LightGBM",
        LGBMClassifier(
            n_estimators=100,         # Boosting iterasyon sayısı
            random_state=42,          # Sabit sonuçlar için
            class_weight='balanced'   # Sınıf dengesizliğiyle otomatik mücadele
        )
    ),
    (
        "Karar Ağacı",
        DecisionTreeClassifier(
            random_state=42,          # Rastgeleliğin sabitlenmesi
            class_weight='balanced'   # Dengesiz veri kümelerinde azınlık sınıfın daha iyi öğrenilmesi sağlanır
        )
    ),
    (
        "En Yakın Komşu",
        KNeighborsClassifier(
            n_neighbors=5             # Her örnek için en yakın 5 komşunun sınıfı dikkate alınır (default=5)
            # Dengesizlik durumlarında performansı sınırlı olabilir; ölçekleme gerektirir
        )
    ),
    (
        "Lojistik Regresyon",
        LogisticRegression(
            max_iter=1000,            # Daha karmaşık veri setlerinde algoritmanın konverge olmasını sağlar
            class_weight='balanced',  # Dengesiz sınıflar için dengeleme yapılır
            random_state=42           # Tekrarlanabilirlik için
        )
    ),
    (
        "Doğrusal SVM",
        LinearSVC(
            max_iter=1000,            # Karmaşık veri setleri için konverjans süresi uzatılır
            class_weight='balanced',  # Sınıf ağırlıkları otomatik hesaplanır
            random_state=42           # Rastgelelik kontrolü
        )
    ),
]

# Modelleri sırayla eğit ve performanslarını yazdır
for model_adi, model in modeller:
    print(f"\n=== {model_adi} Sonuçları ===")
    
    # LightGBM ve XGBoost sadece sayısal etiketlerle çalışabildiğinden özel işlem yapılır
    if model_adi in ["XGBoost", "LightGBM"]:
        model.fit(X_egitim, y_egitim_sayisal)
        tahmin_sayisal = model.predict(X_test)
        tahmin = etiket_donusturucu.inverse_transform(tahmin_sayisal)  # Tahminleri tekrar string hale çevir
    else:
        model.fit(X_egitim, y_egitim)
        tahmin = model.predict(X_test)

    # Her modelin precision, recall, f1-score ve support değerlerini verir
    print(classification_report(y_test, tahmin, digits=3))  # digits=3 → metrikleri 3 basamakla göster
