# CIFAR-10 Image Classification: Machine Learning and Deep Learning Approaches

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nepatiess/aygaz_yapay_zeka/blob/main/yapay_zeka.ipynb)

This project aims to compare the performance of various classification algorithms using the popular **CIFAR-10** image dataset. Within the scope of the project, models were trained using both traditional Machine Learning methods and Artificial Neural Networks (ANN / CNN), and the results were visualized using Confusion Matrices.

---

## 📋 Table of Contents
- [About the Dataset](#about-the-dataset)
- [Models Used](#models-used)
- [Project Workflow](#project-workflow)
- [Requirements](#requirements)
- [How to Run?](#how-to-run)

---

## 🗂 About the Dataset
The **CIFAR-10** dataset consists of a total of 60,000 32x32 pixel color images belonging to 10 different classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). 
- **Training Set:** 50,000 images
- **Test Set:** 10,000 images

---

## 🤖 Models Used
Four different classification algorithms were used in the project:
1. **K-Nearest Neighbors (KNN)**
2. **Random Forest Classifier**
3. **Decision Tree Classifier**
4. **Artificial Neural Network (ANN) / Convolutional Neural Network (CNN)**
   - *Architecture:* 3 `Conv2D` + `MaxPooling2D` layers, regularization with `Dropout`, and 2 `Dense` (Fully Connected) layers.
   - *Optimization:* Adam Optimizer, Categorical Crossentropy loss function.

---

## ⚙️ Project Workflow
1. **Importing Libraries:** Necessary Python libraries are included in the project.
2. **Data Loading and Exploration:** The CIFAR-10 dataset is loaded via Keras, and 25 random images are displayed along with their labels.
3. **Data Preprocessing:** 
   - Image pixels are normalized from the 0-255 range to the 0-1 range (`X / 255.0`).
   - For traditional ML algorithms (KNN, RF, DT), 3D matrices are converted into 1D arrays (flattened).
   - For the ANN model, labels (y) are converted to One-Hot Encoding format (`to_categorical`).
4. **Model Training and Prediction:** The 4 specified models are trained, and predictions are made on the test set.
5. **Evaluation:** The performance of each model is visualized using Confusion Matrices created with the `Seaborn` library.

---

## 🛠 Requirements
For this project to run on your local machine, the following libraries must be installed:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

---

## 🚀 How to Run?
1. Option 1: **On Google Colab**
- You can run the project directly from your browser without any installation by clicking the "Open in Colab" button above.
2. Option 2 : **On a Local Machine**
  - Clone the repository to your computer:
    ```
    git clone [https://github.com/nepatiess/aygaz_yapay_zeka.git (https://github.com/nepatiess/aygaz_yapay_zeka.git)
    ```
  - Navigate to the project directory and start Jupyter Notebook:
  ```
  cd aygaz_yapay_zeka
  jupyter notebook yapay_zeka.ipynb
  ```
  - Run the cells sequentially and examine the results.
 
---

  **NOTE:** *Traditional machine learning models (KNN, Random Forest, etc.) may run slower and yield lower accuracy rates on image data (especially when pixels are flattened) compared to Deep Learning (CNN) models. This project is a great example to observe this difference.*


---

# CIFAR-10 Görüntü Sınıflandırma: Makine Öğrenmesi ve Derin Öğrenme Yaklaşımları

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nepatiess/aygaz_yapay_zeka/blob/main/yapay_zeka.ipynb)

Bu proje, popüler **CIFAR-10** görüntü veri setini kullanarak çeşitli sınıflandırma algoritmalarının performansını karşılaştırmayı amaçlamaktadır. Proje kapsamında hem geleneksel Makine Öğrenmesi (Machine Learning) yöntemleri hem de Yapay Sinir Ağları (YSA / CNN) kullanılarak modeller eğitilmiş ve sonuçlar Karmaşıklık Matrisleri (Confusion Matrix) ile görselleştirilmiştir.

---

## 📋 İçindekiler
- [Veri Seti Hakkında](#veri-seti-hakkında)
- [Kullanılan Modeller](#kullanılan-modeller)
- [Proje İş Akışı](#proje-iş-akışı)
- [Gereksinimler](#gereksinimler)
- [Nasıl Çalıştırılır?](#nasıl-çalıştırılır)

---

## 🗂 Veri Seti Hakkında
**CIFAR-10** veri seti, 10 farklı sınıfa ait (uçak, otomobil, kuş, kedi, geyik, köpek, kurbağa, at, gemi, kamyon) toplam 60.000 adet 32x32 piksel renkli görüntüden oluşmaktadır. 
- **Eğitim Seti:** 50.000 görüntü
- **Test Seti:** 10.000 görüntü

---

## 🤖 Kullanılan Modeller
Projede 4 farklı sınıflandırma algoritması kullanılmıştır:
1. **K-En Yakın Komşu (K-Nearest Neighbors - KNN)**
2. **Rastgele Orman (Random Forest Classifier)**
3. **Karar Ağacı (Decision Tree Classifier)**
4. **Yapay Sinir Ağı (YSA) / Evrişimli Sinir Ağı (CNN)**
   - *Mimari:* 3 adet `Conv2D` + `MaxPooling2D` katmanı, `Dropout` ile regülarizasyon ve 2 adet `Dense` (Tam Bağlantılı) katman.
   - *Optimizasyon:* Adam Optimizer, Categorical Crossentropy kayıp fonksiyonu.

---

## ⚙️ Proje İş Akışı
1. **Kütüphanelerin İçe Aktarılması:** Gerekli Python kütüphaneleri projeye dahil edilir.
2. **Veri Yükleme ve Keşif:** CIFAR-10 veri seti Keras üzerinden yüklenir ve rastgele 25 görüntü etiketleriyle birlikte ekrana yazdırılır.
3. **Veri Ön İşleme:** 
   - Görüntü pikselleri 0-255 aralığından 0-1 aralığına normalize edilir (`X / 255.0`).
   - Geleneksel ML algoritmaları (KNN, RF, DT) için 3 boyutlu matrisler tek boyutlu dizilere (flatten) dönüştürülür.
   - YSA modeli için etiketler (y) One-Hot Encoding formatına (`to_categorical`) dönüştürülür.
4. **Model Eğitimi ve Tahmin:** Belirtilen 4 model eğitilir ve test seti üzerinde tahminler yapılır.
5. **Değerlendirme:** Her bir modelin performansı, `Seaborn` kütüphanesi kullanılarak oluşturulan Karmaşıklık Matrisleri (Confusion Matrix) ile görselleştirilir.

---

## 🛠 Gereksinimler
Bu projenin yerel bilgisayarınızda çalışması için aşağıdaki kütüphanelerin yüklü olması gerekmektedir:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

---

## 🚀 Nasıl Çalıştırılır?
1. Seçenek 1: **Google Colab Üzerinde**
- Yukarıdaki "Open in Colab" butonuna tıklayarak projeyi doğrudan tarayıcınız üzerinden hiçbir kurulum yapmadan çalıştırabilirsiniz.
2. Seçenek 2 : **Yerel Makinede**
  - Repoyu bilgisayarınıza klonlayın:
    ```
    git clone [https://github.com/nepatiess/aygaz_yapay_zeka.git (https://github.com/nepatiess/aygaz_yapay_zeka.git)
    ```
  - Proje dizinine gidin ve Jupyter Notebook'u başlatın:
  ```
  cd aygaz_yapay_zeka
  jupyter notebook yapay_zeka.ipynb
  ```
  - Hücreleri sırasıyla çalıştırarak sonuçları inceleyin.
 
---

  **NOT:** *Geleneksel makine öğrenmesi modelleri (KNN, Random Forest vb.) görüntü verilerinde (özellikle pikseller düzleştirildiğinde - flattened) Derin Öğrenme (CNN) modellerine kıyasla daha yavaş çalışabilir ve daha düşük doğruluk oranları verebilir. Bu proje, bu farkı gözlemlemek için harika bir örnektir.*
