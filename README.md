# 🌲 Kablosuz Sensör Ağlarında Orman Yangını Algılama ve Ağ İçi Veri Birleştirme (In-Network Data Aggregation) Simülasyonu

[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Library](https://img.shields.io/badge/Pygame-2.0+-green?style=for-the-badge&logo=pygame)](https://www.pygame.org/)
[![Library](https://img.shields.io/badge/SimPy-4.0+-orange?style=for-the-badge)](https://simpy.readthedocs.io/)
[![License](https://img.shields.io/badge/Academic-Project-red?style=for-the-badge)](#)

Bu proje, Kablosuz Sensör Ağları (WSN - Wireless Sensor Networks) üzerinde orman yangını senaryosunu temel alarak **Ağ İçi Veri Birleştirme (In-Network Data Aggregation)** tekniklerinin ağ ömrü, enerji verimliliği ve veri trafiği üzerindeki etkilerini inceleyen bilimsel ve görsel bir simülasyon çalışmasıdır.

---

## 🎓 Akademik Bilgiler & Teşekkür

Bu proje, değerli akademisyenimiz **Dr. Öğr. Üyesi HASAN SERDAR** tarafından verilen ödev kapsamında geliştirilmiştir. Projenin akademik derinliği, WSN'lerde yönlendirme protokolleri (TAG, LEACH) ve veri birleştirme teorisi üzerine temellendirilmiştir.

* **Ödevi Veren:** Dr. Öğr. Üyesi HASAN SERDAR
* **Geliştirici (Öğrenci):** [@JunnierDeveloper](https://github.com/JunnierDeveloper)

---

## 📖 Teorik Altyapı ve Proje Amacı

Kablosuz Sensör Ağlarında (WSN) düğümler genellikle sınırlı batarya kapasitesine (enerjiye) sahiptir. Bir sensör düğümünün en çok enerji tükettiği işlem **veri iletimidir (Radyo TX)**. Literatürde (örneğin *TinyDB* ve *TAG - Tiny Integration Service* protokollerinde) tüm uç düğümlerin verileri doğrudan merkeze (Sink/Base Station) göndermesi yerine, ağın belirli bölgelerinde konumlanan **Küme Başları (Cluster Head - CH)** üzerinden verilerin toplanıp sıkıştırılması veya matematiksel olarak birleştirilmesi (Aggregation - örn: MAX, MIN, AVERAGE) önerilir.

Bu simülasyonda:
1. **Ham İletim (Raw Transmission):** Birleştirme olmadan tüm yangın verilerinin doğrudan merkeze aktarılması senaryosu.
2. **Birleştirilmiş İletim (Aggregated Transmission):** Sensörlerin veriyi en yakın Küme Başına (CH) ilettiği, CH'nin ise bu verileri **Ağ İçi Veri Birleştirme** işlemine tabi tutarak tek bir paket halinde merkeze ilettiği senaryo.

Bu iki senaryonun **Trafik Azaltma Oranı** ve **Enerji Tasarrufu (milliJoule - mJ)** parametreleri gerçek zamanlı olarak karşılaştırılmaktadır.

---

## 📐 Enerji ve Tüketim Modeli

Simülasyonda kullanılan bilimsel enerji harcama modeli parametreleri (milliJoule - mJ cinsinden):

| Parametre | Değer | Açıklama |
| :--- | :--- | :--- |
| `INITIAL_ENERGY` | 100.0 mJ | Sensör düğümlerinin başlangıç batarya kapasitesi |
| `E_SENSE` | 0.5 mJ | Çevreden veri okuma/ölçüm yapma maliyeti |
| `E_TX` | 2.0 mJ | Bir paketi radyo kanalıyla gönderme maliyeti |
| `E_RX` | 1.0 mJ | Bir paketi radyo kanalıyla alma maliyeti |
| `E_AGG` | 1.5 mJ | Küme Başında veri birleştirme (CPU işleme) maliyeti |

---

## 🏗️ Sistem Mimarisi

Aşağıdaki şemada uç sensörler, Küme Başları (CH) ve Ana İstasyon (Base Station) arasındaki dinamik veri akışı gösterilmektedir:

```mermaid
graph TD
    subgraph Yangın Bölgesi (Isı > 45°C)
        S1[Sensör 1] -- "Ham Veri (E_TX)" --> CH1((Küme Başı 1 - CH))
        S2[Sensör 2] -- "Ham Veri (E_TX)" --> CH1
        S3[Sensör 3] -- "Ham Veri (E_TX)" --> CH1
    end

    subgraph Güvenli Bölge (Isı <= 45°C)
        S4[Sensör 4] -. "Ölçüm Yapar (E_SENSE) <br> İletim Yapmaz" .-> CH2((Küme Başı 2 - CH))
    end

    CH1 -- "Ağ İçi Birleştirilmiş Paket <br> (MAX Sıcaklık)" --> BS[Base Station / Ana İstasyon]
    CH2 -- "Veri Yoksa Boş Paket" --> BS

    style CH1 fill:#f96,stroke:#333,stroke-width:2px
    style CH2 fill:#9cf,stroke:#333,stroke-width:2px
    style BS fill:#8f8,stroke:#333,stroke-width:4px
    style S1 fill:#ff9,stroke:#333
    style S2 fill:#ff9,stroke:#333
    style S3 fill:#ff9,stroke:#333
```

---

## 🚀 Öne Çıkan Özellikler

Proje iki farklı Python arayüz sürümü içermektedir:

### 1. 📊 Matplotlib Tabanlı Analiz (`aggregation_sim.py`)
* **SimPy Motoru:** Ayrık olay simülasyonu ile düğüm durumları ve zaman senkronizasyonu.
* **Isı Yayılım Modeli:** Rüzgar vektörü yönünde (`WIND_VECTOR`) zamanla Gauss tabanlı yayılan dinamik yangın cephesi.
* **Dashboard Paneli:** Eş zamanlı olarak ağ ömrü, can kaybı (ölü sensörler), ham paket sayıları, birleştirilmiş paket sayıları ve enerji tüketim grafiklerinin analitik gösterimi.

### 2. 🎮 Pygame Tabanlı Gelişmiş Görselleştirme (`aggregation_sim_pygame.py`)
* **Vektörel Isı Haritası:** NumPy `surfarray` kullanılarak ekran kartı düzeyinde hızlandırılmış, sıcaklığa bağlı pürüzsüz geçişli (soğuktan sıcağa HSL) zemin kaplaması.
* **Partikül Sistemi (Particle System):** Yangın bölgelerinden dinamik olarak yükselen, rüzgar yönüne göre savrulan ateş ve duman partikülleri.
* **İletim Animasyonları:** Veri iletildiğinde sensörden Küme Başına çizilen ışık hatları ve yayılan **Ping Halkası** animasyonları.
* **Ağaç Şablonlu Sensör Görselleri:** Canlı ağaçlar (yeşil), yanan ağaçlar (turuncu/kırmızı) ve kuruyup ölen sensörler (gri kuru dallar) şeklinde özelleştirilmiş pikselsel gösterim.
* **Rüzgar Pusulası:** Rüzgarın estiği yönü gerçek zamanlı gösteren vektörel pusula göstergesi.
* **Kaydırılabilir (Scroll) Sağ Panel:** Fare tekerleğiyle aşağı/yukarı kaydırılabilen; her CH'nin anlık can/bağlantı durumlarını, ağ trafiğindeki tasarruf oranını ve sembol açıklamalarını gösteren zengin gösterge tablosu.

---

## 🛠️ Kurulum ve Çalıştırma

### 1. Sanal Ortam Oluşturma ve Aktifleştirme
Proje dizininde kütüphane çakışmalarını önlemek amacıyla sanal ortam kurulması önerilir.

```bash
# Proje klasörüne gidin
cd "Orman Yangını Veri Birleştirme Similasyonu"

# Sanal ortamı oluşturun
python -m venv env
```

**Sanal Ortamı Aktif Edin:**
* **Windows (PowerShell):**
  ```powershell
  .\env\Scripts\Activate.ps1
  ```
* **Windows (CMD / Komut Satırı):**
  ```cmd
  env\Scripts\activate
  ```
* **macOS / Linux:**
  ```bash
  source env/bin/activate
  ```

### 2. Gerekli Kütüphanelerin Yüklenmesi
Gerekli bilimsel ve görsel kütüphaneleri yüklemek için:

```bash
pip install -r requirements.txt
# veya manuel olarak:
pip install simpy numpy pygame matplotlib
```

### 3. Simülasyonu Başlatma

* **Gelişmiş Pygame Simülasyonunu Çalıştırmak İçin:**
  ```bash
  python aggregation_sim_pygame.py
  ```
  *(Pygame arayüzündeyken **SPACE (Boşluk)** tuşuna basarak simülasyonu duraklatabilir veya sağ üstteki **DURDUR / BAŞLAT** butonunu kullanabilirsiniz.)*

* **Matplotlib Analiz Sürümünü Çalıştırmak İçin:**
  ```bash
  python aggregation_sim.py
  ```

---

## 📝 Simülasyon Sonuçlarının Yorumlanması

Simülasyon tamamlandığında veya çalışma esnasında elde edilen çıktılar şu akademik bulguları kanıtlar:
* **Trafik Düşüşü:** Ağ içi veri birleştirme (Aggregation) sayesinde iletilen paket sayısı yaklaşık **%70 - %85 oranında azalır**. Bu durum ağdaki veri çarpışmalarını (collision) ve darboğazları engeller.
* **Enerji Tasarrufu:** Paket sayısının azalması, sensörlerin TX radyolarını çok daha az kullanmasını sağlayarak ağın toplam enerji tüketiminde ciddi bir tasarruf sağlar ve ağın ömrünü (lifetime) katlarca uzatır.
