# Akıllı Doküman Arama ve Soru-Cevap Sistemi (RAG)

Bu proje, yerel bilgisayarınızdaki PDF, TXT ve Markdown dosyalarını analiz ederek, bu dokümanlardaki bilgilere göre sorularınızı yanıtlayan hafif ve modüler bir **RAG (Retrieval-Augmented Generation)** sistemidir.

Sistemin en büyük özelliği; doküman parçalama ve benzerlik araması (indeksleme) işlemlerini hiçbir dış kütüphane veya veri tabanına ihtiyaç duymadan **saf Python ve Numpy kullanarak (TF-IDF & Cosine Similarity ile)** yerel olarak gerçekleştirmesidir.

## Proje Yapısı

*   `main.py`: Kullanıcıyla etkileşime geçen interaktif komut satırı (CLI) arayüzüdür. Dokümanları yükler, indeksleme döngüsünü başlatır ve soruları alır.
*   `pdf_parser.py`: PDF ve TXT dosyalarını okuyarak metne dönüştürür. Hangi bilginin hangi sayfada olduğunu takip etmek için sayfa numaralarını korur.
*   `retriever.py`: Dokümanları belirlenen boyutlarda parçalar, kelime haznesi çıkarıp TF-IDF indeks matrisi oluşturur. Arama yapıldığında Cosine Similarity kullanarak en alakalı paragrafları bulur.
*   `llm_client.py`: En alakalı bulunan paragrafları (context) alıp Gemini API'sine (`gemini-2.5-flash`) göndererek sentezlenmiş, doğru ve kaynak belirten cevaplar üretir.
*   `config.json`: Gemini API anahtarı, taranacak doküman klasör adı ve parça ayarlarını barındırır.
*   `docs/`: Sorgulamak istediğiniz dosyaları koyacağınız klasör.

## Kurulum ve Bağımlılıklar

Gerekli olan tek harici kütüphane PDF'leri okumak için kullanılan `pypdf` kütüphanesidir:

```bash
pip install pypdf requests numpy
```

## Gemini API Anahtarı Alma

Sistemin yapay zeka cevapları üretebilmesi için ücretsiz bir Gemini API anahtarı alabilirsiniz:
1.  [Google AI Studio](https://aistudio.google.com/) adresine gidin.
2.  Google hesabınızla giriş yapın.
3.  **Get API Key** butonuna tıklayarak yeni bir anahtar oluşturun.
4.  Oluşturduğunuz anahtarı kopyalayıp projedeki `config.json` dosyasında yer alan `"gemini_api_key"` değerine yapıştırın.

> [!TIP]
> **Çevrimdışı Çalışma (Offline Mode):**
> Eğer `config.json` dosyasında bir API anahtarı tanımlamazsanız, sistem hata verip kapanmaz. Çevrimdışı modda çalışarak arattığınız soruyla en alakalı doküman paragraflarını kaynak dosyası ve sayfa numarası ile birlikte doğrudan terminale listeler.

## Nasıl Çalıştırılır?

1.  Varsayılan olarak proje klasörünün içindeki `docs/` klasörüne analiz etmek istediğiniz PDF/TXT dosyalarını koyun. (Eğer `docs/` klasörü boşsa, sistem otomatik olarak bir üst klasördeki PDF/TXT dosyalarını bulup indekslemeyi teklif edecektir).
2.  Projeyi başlatın:

```bash
python main.py
```

### Kullanım Komutları:
*   `Soru > `: Dokümanlardaki herhangi bir bilgiyi sorun (Örn: *YOLOv8 Jetson kurulum adımları nelerdir?* veya *İDA teknik şartnamesindeki maksimum ağırlık sınırı nedir?*)
*   `list`: Şu anda hafızada indekslenmiş olan tüm doküman kaynaklarını listeler.
*   `q` veya `cikis`: Programı sonlandırır.
