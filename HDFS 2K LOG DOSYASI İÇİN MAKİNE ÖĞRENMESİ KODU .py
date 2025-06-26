import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier


# Bu veri kümesinde her satır bir BlockId'ye karşılık gelen EventId sayımlarını ve etiketini içerir.
veri = pd.read_csv("C:/Users/atkn_/Desktop/AAA/features_hdfs_labeled.csv") # Bu kısım etiketli özellik vektörlerini CSV dosyasından okur

# Özellikler (X) ve etiketler (y) olarak veriyi ayırıyoruz.
# Etiket: "label" sütunu — Normal (0) veya Anomali (1)
# Özellikler: "BlockId" ve "label" dışındaki tüm sütunlardır.
ozellikler = veri.drop(columns=["label", "BlockId"], errors="ignore")  # features = X
etiketler = veri["label"]  # target = y

# Eğer etiketler metin (string) formatındaysa sayısal formata dönüştürmemiz gerekiyor o yüzden koda bu kısmı ekliyoruz.
etiketler = etiketler.map({"Normal": 0, "Anomaly": 1})

# Eğitim ve test verisini ayırıyoruz.
# %80 eğitim, %20 test oranıyla rastgele ancak sınıf oranlarını koruyacak şekilde bölüyoruz.
ozellikler_egitim, ozellikler_test, etiketler_egitim, etiketler_test = train_test_split(
    ozellikler, etiketler, test_size=0.2, random_state=42, stratify=etiketler
)

# Kullanılacak sınıflandırma modellerini tanımlıyoruz.
# Logistic Regression, Random Forest, SVM ve XGBoost modelleri karşılaştırılacaktır
modeller = {
    "Lojistik Regresyon": LogisticRegression(max_iter=500, class_weight='balanced'),
    "Rasgele Orman": RandomForestClassifier(class_weight='balanced'),
    "Destek Vektör Makinesi (SVM)": SVC(class_weight='balanced'),
    "XGBoost": XGBClassifier(eval_metric='logloss')  # logloss: binary sınıflama için uygun değerlendirme metriği
}

# Her bir modeli sırayla eğitiyoruz ve test seti üzerinde değerlendiriyoruz.
for model_adi, model in modeller.items():
    model.fit(ozellikler_egitim, etiketler_egitim)  # Modeli eğitim verisiyle eğitiyoruz.
    tahminler = model.predict(ozellikler_test)      # Test verisi üzerinde tahmin yapıyoruz.
    
    # Doğruluk (accuracy) ve F1 skoru hesaplıyoruz.
    dogruluk = accuracy_score(etiketler_test, tahminler)
    f1 = f1_score(etiketler_test, tahminler, pos_label=1, zero_division=0)
    
    # Sonuçları ekrana yazdırıyoruz.
    print(f"\n🔹 {model_adi}")
    print(f"   ✅ Doğruluk (Accuracy): {dogruluk:.4f} | F1 Skoru: {f1:.4f}")
    
    # Sınıf bazlı precision, recall ve F1 skoru raporu
    print(classification_report(
        etiketler_test,
        tahminler,
        target_names=["Normal", "Anomali"],
        zero_division=0
    ))
