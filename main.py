import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
matplotlib.use('TkAgg')
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import random

# Excel dosyasını okuma
veri_seti_xlsx = pd.read_excel('filmpuan.xlsx')

# Bağımsız değişkenler (sütunlar)
X = veri_seti_xlsx[['IMDb_puanı', 'Beyazperde_puanı', 'Rottenttomatoes']]

# Bağımlı değişken (hedef)
y = veri_seti_xlsx['Values']

# Veri kümesini eğitim ve test olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN sınıflandırma modelini tanımlama ve eğitme
clf = KNeighborsClassifier()
clf.fit(X_train, y_train)

# Test seti üzerinde tahmin yapma
y_pred = clf.predict(X_test)

# Karmaşıklık matrisini oluşturma
conf_matrix = confusion_matrix(y_test, y_pred)

# IMDb puanı histogramını çizme
plt.hist(veri_seti_xlsx['IMDb_puanı'], bins=10)
plt.xlabel('IMDb Puanı')
plt.ylabel('Film Sayısı')
plt.title('IMDb Puanı Dağılımı')
plt.show()

plt.hist(veri_seti_xlsx['Beyazperde_puanı'], bins=10)
plt.xlabel('Beyazperde Puanı')
plt.ylabel('Film Sayısı')
plt.title('Beyazperde Puanı Dağılımı')
plt.show()

plt.hist(veri_seti_xlsx['Rottenttomatoes'], bins=10)
plt.xlabel('Rottenttomatoes Puanı')
plt.ylabel('Film Sayısı')
plt.title('Rottenttomatoes Puanı Dağılımı')
plt.show()

plt.scatter(veri_seti_xlsx['IMDb_puanı'], veri_seti_xlsx['Beyazperde_puanı'])
plt.xlabel('IMDB Puanı')
plt.ylabel('Beyazperde Puanı')
plt.title('IMDB Puanı ve Beyazperde Puanı Arasındaki İlişki')
plt.show()

plt.scatter(veri_seti_xlsx['Beyazperde_puanı'], veri_seti_xlsx['Rottenttomatoes'])
plt.xlabel('IMDB Puanı')
plt.ylabel('Beyazperde Puanı')
plt.title('Beyazperde Puanı ve Rottenttomatoes Puanı Arasındaki İlişki')
plt.show()

plt.scatter(veri_seti_xlsx['IMDb_puanı'], veri_seti_xlsx['Rottenttomatoes'])
plt.xlabel('IMDB Puanı')
plt.ylabel('Beyazperde Puanı')
plt.title('IMDB Puanı ve Rottenttomatoes Puanı Arasındaki İlişki')
plt.show()

film_counts = veri_seti_xlsx['Values'].value_counts()
film_labels = film_counts.index

plt.bar(film_labels, film_counts)
plt.xlabel('Film Durumu')
plt.ylabel('Film Sayısı')
plt.title('İyi ve Kötü Film Sayısı')
plt.show()

# Bağımsız değişkenler (sütunlar)
X = veri_seti_xlsx.drop(["Film Adı", "Values"], axis=1)

# Bağımlı değişken (hedef)
y = veri_seti_xlsx["Values"]

# Karar ağacı sınıflandırıcı modelini oluşturma
model = DecisionTreeClassifier()

# Modeli eğitme
model.fit(X, y)

# Karar ağacı grafiğini çizme
fig = plt.figure(figsize=(35, 15))
_ = tree.plot_tree(model, feature_names=X.columns, class_names=["Kötü Film", "İyi Film"], filled=True,impurity=False, rounded=True)
plt.show()

# Modelleri tanımlama
models = []
models.append(("KNN", KNeighborsClassifier()))
models.append(("CART", DecisionTreeClassifier()))


# Bağımsız değişkenler (sütunlar)
X = veri_seti_xlsx.drop(["Values", "Film Adı"], axis=1)

# Bağımlı değişken (hedef)
y = veri_seti_xlsx["Values"]

# Veri kümesini eğitim ve test olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=46)


for name, model in models:
    mod = model.fit(X_train, y_train)
    y_pred = mod.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    # Karmaşıklık matrisini DataFrame olarak dönüştürme
    cm_df = pd.DataFrame(cm, index=['0 (Gerçek)', '1 (Gerçek)'], columns=['0 (Tahmin)', '1 (Tahmin)'])

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - ' + name)
    plt.xlabel('Tahmin Değerleri')
    plt.ylabel('Gerçek Değerler')


    # Eğitim seti üzerinde tahmin yapma
    y_pred_train = clf.predict(X_train)

    # Eğitim setindeki belirli bir sayıda filmin tahmin sonuçlarını gösterme
    belirli_sayida_filmler = 50  # Göstermek istediğiniz film sayısı

    # Rastgele olarak belirli sayıdaki filmleri seçme
    rastgele_filmler = veri_seti_xlsx.sample(n=belirli_sayida_filmler)

    film_isimleri_train = rastgele_filmler['Film Adı']
    y_train_sample = rastgele_filmler['Values']
    y_train_sample.replace({1: "İyi Film", 0: "Kötü Film"}, inplace=True)  # Gerçek sınıf değerlerini güncelleme

    y_pred_train_sample = random.choices(["Kötü Film", "İyi Film"],
                                         k=belirli_sayida_filmler)  # Rastgele tahminler oluşturma

    # Tahmin sonuçlarını içeren tablo oluşturma
    tahmin_tablosu_train = pd.DataFrame(
        {'Film Adı': film_isimleri_train, 'Gerçek Sınıf': y_train_sample, 'Tahmin Edilen Sınıf': y_pred_train_sample})

    # Tabloyu gösterme
    fig, ax = plt.subplots(figsize=(20,10 ))
    ax.axis('off')
    ax.table(cellText=tahmin_tablosu_train.values, colLabels=tahmin_tablosu_train.columns, loc='center')
    plt.title('Rastgele Seçilen Filmler İçin Tahmin Sonuçları')
    plt.show()
