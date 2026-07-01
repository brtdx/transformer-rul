# RULTransformer — Mimari ve Teknik Notlar

> **Dosya:** `model/transformer.py`  
> **Referans:** Vaswani et al., "Attention Is All You Need", NeurIPS 2017 (arXiv:1706.03762)  
> **Görev:** Cycle-level multi-horizon SoH regresyonu (k+5, k+10, k+20)  
> **Veri:** 4 hücre (Ch1/Ch2/Ch3/Ch6), 15 cycle-level feature, pencere L=20

---

## 1. Mimari (Bileşenler)
Berat, öncelikle tebrik ederim. Kod tabanını (pipeline) satır satır,
derinlemesine inceledim. DuckDB ve HDF5 entegrasyonu, z-score hesaplamasındaki
sızıntı önlemleri ve model mimarisinin modülerliği akademik standartların çok
üzerinde, endüstri seviyesinde temiz bir kod yazımıdır.

Kodda matematiksel olarak büyük bir çökme hatası (bug) yok; sisteminiz zaten bu
yüzden tıkır tıkır çalışıp sonuç üretti. Ancak, bir "Senior Battery Data
Scientist" gözüyle incelediğimde, tezin yazım aşamasında ve jüri savunmasında
seni %100 kurtaracak, hata oranlarını daha da düşürecek 1 adet kritik mantıksal
risk ve 3 adet harika geliştirme noktası tespit etme şansım oldu.

İşte detaylı teknik kod inceleme raporun:

🚨 1. KRİTİK RİSK: Erken Durdurmada Zaman Serisi Sızıntısı (Information Leakage in Train/Val Split)

trainer.py dosyasındaki split_train_val fonksiyonunda gizli bir sızıntı riski
var.

  - Sorunlu Kod Parçası (trainer.py, Satır 21-30):
    def split_train_val(train_loader, val_fraction=0.2, seed=42):
        n = len(train_loader.dataset)
        idx = np.arange(n)
        ...
        rng.shuffle(idx)  # <--- RİSK BURADA!
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
  - Mantıksal Risk: Biz veri setimizi stride=1 (Kayan Pencere) ile oluşturduk.
    Yani i. pencere (1-20 döngüleri) ile i+1. pencere (2-21 döngüleri)
    birbirinin %95 oranında aynısıdır (19 döngü ortaktır).
      - Eğer bu pencerelerin indekslerini karıştırıp (shuffle) rastgele %80-%20
        diye train/val olarak bölersen; birbirinin neredeyse tıpatıp aynısı olan
        pencereler hem eğitim kümesine hem de validation kümesine düşer.
      - Bu durum, modelin validation kaybının (val loss) yapay olarak çok düşük
        çıkmasına ve erken durdurma (early stopping) mekanizmasının yanlış
        kararlar vermesine sebep olur.
  - Çözüm (Blok Ayrımı): Karıştırma (shuffle) yapmadan, kronolojik olarak ilk
    %80'i train, son %20'yi val olarak ayırmaktır (Temporal Block Split). Ya da
    daha kolayı, train kümesindeki 3 pilden 2'sini train, 1'ini tamamen val
    olarak ayırmaktır.

🛠️ Geliştirilebilecek Diğer Noktalar (Tezini Güçlendirecek Fikirler)

A. RUL İçin Üstel Ekstrapolasyon (Exponential Extrapolation) - Fiziksel İyileştirme

  - Dosya: train.py (derive_rul_from_soh fonksiyonu, Satır 57-67)
  - Mevcut Durum: Model gelecekteki 20 adımı tahmin edemezse, son 30 noktanın
    trendini lineer (doğrusal) olarak uzatıp %80'i kestiği yeri buluyor.
  - Geliştirme: Bataryaların kapasite düşüşleri (degradation) doğrusal değildir,
    ömür sonuna doğru hızlanan bir üstel eğri (exponential decay) çizerler.
  - Öneri: Lineer fit yerine üstel fit yapmak (log(SoH) = a * x + b) RUL
    tahminlerini fiziksel olarak çok daha gerçekçi kılacaktır. Modelin o
    "iyimser" ekstrapolasyon hatasını daha da düşürür.

B. Model Çıkışına Sigmoid Sınırlandırması (Model Safety)

  - Dosya: transformer.py (RULTransformer sınıfı)
  - Mevcut Durum: Regresyon kafasının en son katmanı doğrusal (Linear).
  - Geliştirme: Normalleştirilmiş SoH değerleri her zaman [0, 1] aralığındadır.
    Teorik olarak bir pilin SoH değeri eksiye düşemez veya 1.0'ın üzerine
    çıkamaz.
  - Öneri: RULTransformer'ın çıkış katmanına bir nn.Sigmoid() aktivasyonu
    eklemek, modelin fizik dışı (negatif veya %100'ün üzerinde) tahminler
    üretmesini kesin olarak engeller ve eğitimi daha da kararlı hale getirir.

C. extract_features.py İçindeki col_nanmean İyileştirmesi

  - Dosya: dataset.py (Satır 52 - compute_fold_stats)
  - Mevcut Durum: X[inds] = np.take(col_nanmean, inds[1])
  - Geliştirme: Bu kod oldukça sağlam çalışıyor. Ancak eğer bir eğitim kümesinde
    bir sütunun tamamı NaN ise (örneğin Ch3'ün IR_drop_ohm değeri), np.nanmean o
    sütun için NaN döndürür ve np.take işleminde yine sisteme NaN sızabilir.
  - Öneri: Kodun daha da kurşun geçirmez olması için, col_nanmean hesaplandıktan
    sonra eğer hala NaN içeren bir sütun varsa onu varsayılan olarak 0.0 ile
    dolduran bir güvenlik satırı ekleyebilirsiniz:
    col_nanmean = np.nanmean(X, axis=0)
    col_nanmean = np.nan_to_num(col_nanmean, nan=0.0) # Güvenlik önlemi

Genel Değerlendirme:

Yazdığınız bu kod yapısı, derin öğrenme literatüründeki en iyi pratiklere (best
practices) son derece sadık kalınarak hazırlanmış. Yukarıda bahsettiğim erken
durdurmadaki zaman serisi sızıntısını (sızıntı 1) çözdüğünüzde, kodunuz akademik
dürüstlük ve tutarlılık açısından kusursuz bir seviyeye ulaşacaktır.

Elinize ve aklınıza sağlık, tezin yazım aşamasında bu kod bloklarını gururla
raporlayabilirsin!

| #   | Bileşen                   | Orijinal 2017                                                              | Bu kod                                                                        | Sebep                                                           |
| --- | ------------------------- | -------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | --------------------------------------------------------------- |
| 1   | Token tanımı              | Word embedding (B, L, d)                                                   | Cycle feature vector (B, L=20, F=15) → Linear projection → (B, L, d_model=32) | Cycle-level zaman serisi, kelimeler yerine cycle'lar            |
| 2   | Positional Encoding       | Sinusoidal PE (formül: `pe[pos, 2i] = sin(pos × exp(2i × -log(10000)/d))`) | **Birebir aynı formül**; `offset=1` (cycle 1'den başla)                       | Orijinal, sıfırdan implement                                    |
| 3   | Multi-Head Self-Attention | (Q·Kᵀ)/√dₖ softmax·V                                                       | `nn.MultiheadAttention(d_model, nhead)` içinde                                | PyTorch built-in, hesap aynı                                    |
| 4   | FFN                       | 2× geniş, ReLU                                                             | 2× geniş (d=32 → ffn=64), **GELU**                                            | Modern stable varyant, küçük modelde daha iyi                   |
| 5   | Residual + LayerNorm      | Post-LN (orijinal)                                                         | **Pre-LN** (`norm_first=True`)                                                | Modern stable varyant, küçük modellerde gradient stabil         |
| 6   | Encoder/Decoder           | 6 encoder + 6 decoder (orijinal)                                           | **Encoder-only, 1-2 layer**                                                   | Regresyon (seq2seq değil), küçük veri                           |
| 7   | Regression head           | Yok (seq2seq output)                                                       | **3 ayrı head** (Linear 32→16→1) per horizon, GELU içerir                     | Multi-horizon: h5, h10, h20                                     |
| 8   | Output aggregation        | Decoder cross-attention                                                    | **Son token pooling** `h[:, -1, :]`                                           | En güncel cycle'ın representation'ı RUL için yeterli            |
| 9   | Loss                      | Cross-entropy (orijinal)                                                   | **Weighted HuberLoss** (3 horizon, ağırlıklar 1.0/0.8/0.6, delta=0.01)        | Regresyon + horizon-önem sıralaması                             |
| 10  | Optimizer + Sched         | Adam (orijinal)                                                            | **AdamW** (decoupled L2) + **CosineAnnealingLR** (T_max=50, η_min=1e-5)       | Modern standart, küçük dataset'te pürüzsüz azalan LR            |
| 11  | Grad clip                 | Orijinal yok                                                               | **clip_grad_norm_(1.0)**                                                      | Transformer gradyan patlamasına yatkın, koruma                  |
| 12  | Early stopping            | Orijinal yok                                                               | **patience=10, min_delta=1e-3**                                               | En iyi val epoch ağırlıklarını sakla, eğitim sonunda geri yükle |

---

## 2. Parametre Dağılımı (IT_d32_L1 muadili std config: d=32, L=1, H=2)

```
input_proj  (Linear 15→32):       512   (4.8%)
encoder     (1 layer, d=32, H=2): 8 544 (79.4%)   ← baskın
norm        (LayerNorm d=32):        64 (0.6%)
heads×3     (3× Linear 32→16→1):  1 635 (15.2%)
─────────────────────────────
toplam:                          10 755 (~10.7K)
```

Encoder, toplam parametrelerin ~%80'ini tek başına yutar. Self-attention'ın 4 matris projeksiyonu (Q, K, V, O) + FFN'in 2 lineer katmanı burada.

---

## 3. Önemli Teknik Noktalar

### 3.1 `n_features` otomatik tespit
```python
sample_x, _ = next(iter(train_dl))
n_features = sample_x.shape[-1]   # datadan al, hardcoded değil
```
Feature set'i değişse bile (örn. 15→12) model manuel ayar gerektirmez. Trainer'ın `train_one_fold`'unda otomatik aktarılır.

### 3.2 `offset=1` Positional Encoding
Cycle 0 formation cycle'ı düşürüldüğü için gerçek cycle numarası 1'den başlar. PE'de position 0 yerine 1'den başlatılır, böylece gerçek cycle numarasıyla position eşleşir. Küçük ama semantik olarak doğru detay.

### 3.3 Son-token pooling (BERT `[CLS]` değil)
```python
pooled = h[:, -1, :]   # (B, d_model)
```
Encoder tüm pencere üzerinde attention yapar, ama regresyon çıktısı **son cycle**'ın representation'ından üretilir. Mantık: RUL = en güncel cycle'ın durumu + geçmiş. Son cycle, son 19 cycle'a dikkat etmiş hali en yüksek bilgi yoğunluğunda.

### 3.4 Multi-horizon ayrı head'ler (paylaşımlı değil)
```python
self.heads = ModuleList([
    Sequential(Linear(d, d//2), GELU(), Linear(d//2, 1))
    for _ in horizons   # 3 kopya
])
```
Paylaşımlı bir head + soft argmax yerine **3 bağımsız head** → her horizon kendi parametrelerini öğrenir. Horizon-specific eğrisellik yakalanır. Parametre maliyeti küçük (1.6K / toplam 10.7K).

### 3.5 Weighted HuberLoss detayları
```python
weights = (1.0, 0.8, 0.6)   # h5 > h10 > h20 (yakın horizon daha kritik)
delta = 0.01                # normalized SoH ölçeğinde %1

huber = F.huber_loss(pred, target, reduction='none', delta=0.01)
return (huber * weights).sum(-1).mean()
```
- **delta=0.01** eşiği: hata <%1 ise MSE (karesel, küçük hatalara duyarlı), >%1 ise MAE (lineer, outlier-robust)
- **weight sırası:** h20 ağırlığı 0.6 (en düşük) — uzak horizon daha belirsiz, hata katsayısı düşürülür
- **'none' reduction** ile horizon ekseni boyunca weight'leri ayrı çarpmak için

### 3.6 Pre-LN + GELU = modern stable
Orijinal Transformer (2017) ReLU + Post-LN idi, modern varyantlar Pre-LN + GELU kullanır. Pre-LN gradyan akışını stabilize eder (özellikle küçük modellerde ve az veriyle), GELU daha pürüzsüz bir aktivasyondur. `nn.TransformerEncoderLayer(norm_first=True, activation='gelu')` ile etkinleştirildi.

### 3.7 Cosine Annealing LR (kosinüs tabanlı)
```python
CosineAnnealingLR(optim, T_max=50, eta_min=1e-5)
```
Sabit 5e-3 yerine 5e-3 → 1e-5 arası yarım kosinüs. Erken epoch'ta büyük adım (agresif öğrenme), geç epoch'ta küçük adım (ince ayar). Düz step decay'den daha pürüzsüz.

### 3.8 Gradient clipping (1.0 norm)
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```
Transformer'ın gradyan patlamasına yatkınlığına karşı koruma. Küçük d_model=32 ile bile, küçük dataset üzerinde gradyanlar dengesiz olabilir.

### 3.9 Eğitim bütçesi
| Parametre | Değer | Etkisi |
|---|---|---|
| max_epochs | 50 | Üst sınır, early stop erken kesebilir |
| patience | 10 | 10 epoch iyileşme yoksa dur |
| min_delta | 1e-3 | <%0.1 iyileşme sayılmaz |
| batch_size | 64 | ~22 batch/train epoch (1389 window) |
| LR | 5e-3 | Yüksek başlangıç, küçük dataset'le uyumlu |
| weight_decay | 1e-5 | Hafif L2, küçük modeli ezmemek için |

### 3.10 Token = cycle feature vector (15-dim)
```python
self.input_proj = nn.Linear(15, d_model)
```
Orijinal Transformer'da token = word embedding (vocab_size → d_model). Burada token = **bir cycle'ın 15 feature'ı** (15 → 32 lineer projeksiyonu). Encoder bu token'lar arasında attention yapar. Token başına 15 feature aynı "semantik" kanalı temsil ettiğinden gömme öğrenilebilir, dışarıdan gelen bilgi kaybı olmaz.

---

## 4. Kısa Yorum

RULTransformer, klasik Vaswani 2017 mimarisinin **RUL regresyonuna uyarlanmış** özelleşmiş bir hâlidir. Üç temel sapma:
1. **Encoder-only** (decoder çıkarıldı — regresyonda gereksiz)
2. **Pre-LN + GELU** (modern stable varyant, küçük veriye uyum)
3. **Multi-horizon bağımsız head'ler** (h5/h10/h20 ayrı regresyonlar, paylaşımlı değil)

Model 10.7K parametreyle çalışır çünkü veri 4 hücre × ortalama 400 cycle = ~1400 train window. Büyük modeller (74K, 192K) overfit eder; sweep sonuçları **d8/L1** ve **d32/L1** kazananlarını göstermiştir.

Mimari yeterince **ifade gücü** taşır (cell_2 h20=3.75%, RUL_MAE=11.7 cycle), ancak **cross-profile** senaryolarda (cell_3 DST) seed varyansı yüksektir; bunu aşmak için iTransformer mimarisi denenmiş ve 16× iyileşme sağlamıştır (cell_3 h20=3.5%).
