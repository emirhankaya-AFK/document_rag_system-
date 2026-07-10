import os
import json
from pdf_parser import DocumentParser
from retriever import DocumentRetriever
from llm_client import GeminiClient

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
PARENT_DIR = os.path.dirname(os.path.dirname(__file__))

def load_config():
    if not os.path.exists(CONFIG_PATH):
        print(f"[ERROR] Config dosyası bulunamadı: {CONFIG_PATH}")
        return {}
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

def get_documents_to_index(docs_dir):
    """
    Scans the docs_dir. If empty, scans the parent directory for PDF/TXT files as a helper.
    """
    valid_exts = ['.pdf', '.txt', '.md', '.log']
    files_to_index = []
    
    # 1. Scan the local docs directory
    if os.path.exists(docs_dir):
        for f in os.listdir(docs_dir):
            file_path = os.path.join(docs_dir, f)
            if os.path.isfile(file_path):
                _, ext = os.path.splitext(f)
                if ext.lower() in valid_exts:
                    files_to_index.append(file_path)
                    
    # 2. Fallback: If empty, check the parent directory
    if not files_to_index:
        print(f"[INFO] '{docs_dir}' dizini boş. Üst dizindeki (`model/`) dokümanlar taranıyor...")
        for f in os.listdir(PARENT_DIR):
            file_path = os.path.join(PARENT_DIR, f)
            if os.path.isfile(file_path):
                _, ext = os.path.splitext(f)
                # Ignore scripts and log files in parent dir, only load documents
                if ext.lower() in ['.pdf', '.txt'] and not f.endswith('.py'):
                    files_to_index.append(file_path)
                    
    return files_to_index

def main():
    print("=== Akıllı Doküman Arama ve Soru-Cevap Sistemi (RAG) ===")
    
    config = load_config()
    if not config:
        return
        
    docs_dir_name = config.get("documents_dir", "docs")
    docs_dir = os.path.join(os.path.dirname(__file__), docs_dir_name)
    
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
        print(f"[INFO] '{docs_dir_name}' klasörü oluşturuldu. Dokümanlarınızı buraya ekleyebilirsiniz.")

    # Initialize components
    retriever = DocumentRetriever(
        chunk_size=config.get("chunk_size", 500),
        chunk_overlap=config.get("chunk_overlap", 100)
    )
    
    # Load and index documents
    files = get_documents_to_index(docs_dir)
    
    if not files:
        print("[WARN] İndekslenecek doküman bulunamadı. Lütfen 'docs/' klasörüne PDF veya TXT dosyaları ekleyin.")
        print("Uygulama sonlandırılıyor.")
        return
        
    print("\n[INFO] Dokümanlar taranıyor ve indeksleniyor...")
    for file_path in files:
        filename = os.path.basename(file_path)
        print(f"-> Yükleniyor: {filename}")
        pages = DocumentParser.load_document(file_path)
        if pages:
            retriever.add_document(file_path, pages)
            
    retriever.build_index()
    
    # Initialize LLM Client
    client = GeminiClient(api_key=config.get("gemini_api_key", ""))
    
    print("\n=== SİSTEM HAZIR ===")
    print("Komutlar:")
    print(" - 'list'  : İndekslenmiş dosyaları listeler.")
    print(" - 'q'     : Programdan çıkış yapar.")
    print("Dokümanlarla ilgili herhangi bir soru sorabilirsiniz...\n")
    
    while True:
        try:
            query = input("Soru > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[INFO] Çıkış yapılıyor...")
            break
            
        if not query:
            continue
            
        if query.lower() in ['q', 'cikis', 'exit']:
            print("[INFO] Görüşmek üzere!")
            break
            
        if query.lower() == 'list':
            unique_sources = sorted(list(set(chunk['source'] for chunk in retriever.chunks)))
            print("\n--- İndekslenmiş Dokümanlar ---")
            for src in unique_sources:
                print(f"- {src}")
            print(f"Toplam {len(retriever.chunks)} paragraf yüklendi.\n")
            continue
            
        # 1. Retrieve relevant chunks
        results = retriever.retrieve(query, top_k=config.get("top_k", 3))
        
        if not results:
            print("\n[Yararlı Bilgi]: Dokümanlarda bu kelimelerle eşleşen bir içerik bulunamadı.\n")
            continue
            
        # 2. Try to get AI Answer
        ai_answer = None
        if client.enabled:
            print("[Düşünüyor...] Gemini API yanıtı bekleniyor...")
            ai_answer = client.generate_answer(query, results)
            
        # 3. Print Results
        if ai_answer:
            print("\n🤖 YAPAY ZEKA CEVABI:")
            print(ai_answer)
            print("\n" + "="*40 + "\n")
        else:
            # Fallback to listing retrieved chunks (Offline Mode or no API key)
            print("\n🔍 YEREL DOKÜMAN ARAMA SONUÇLARI (Alakalı Paragraflar):")
            for i, item in enumerate(results):
                chunk = item['chunk']
                score = item['score']
                print(f"\n[{i+1}] Kaynak: {chunk['source']} (Sayfa: {chunk['page']}) - Skor: {score:.2f}")
                print(f"    {chunk['text']}")
            print("\n" + "="*40 + "\n")

if __name__ == "__main__":
    main()
