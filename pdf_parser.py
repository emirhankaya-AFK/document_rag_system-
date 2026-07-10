import os
from pypdf import PdfReader

class DocumentParser:
    @staticmethod
    def parse_pdf(file_path):
        """
        Parses a PDF file and returns a list of pages: [{'page': i, 'text': text}]
        """
        pages = []
        try:
            reader = PdfReader(file_path)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    pages.append({
                        'page': i + 1,
                        'text': text.strip()
                    })
        except Exception as e:
            print(f"[ERROR] PDF okunurken hata oluştu ({file_path}): {e}")
        return pages

    @staticmethod
    def parse_txt(file_path):
        """
        Parses a TXT file and returns it as page 1.
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().strip()
                if content:
                    return [{'page': 1, 'text': content}]
        except Exception as e:
            print(f"[ERROR] TXT dosyası okunurken hata oluştu ({file_path}): {e}")
        return []

    @classmethod
    def load_document(cls, file_path):
        """
        Loads a document based on extension.
        Returns a list of page dicts.
        """
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext == '.pdf':
            return cls.parse_pdf(file_path)
        elif ext in ['.txt', '.md', '.log']:
            return cls.parse_txt(file_path)
        else:
            print(f"[WARN] Desteklenmeyen dosya biçimi yoksayılıyor: {file_path}")
            return []
