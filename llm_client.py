import requests
import json

class GeminiClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.enabled = True
        
        if not api_key or "YOUR_GEMINI" in api_key:
            print("[WARN] Gemini API Anahtarı bulunamadı veya varsayılan değerde bırakılmış. LLM sentezleme devre dışı.")
            self.enabled = False

    def generate_answer(self, query, context_chunks):
        """
        Generates an answer from Gemini model based on the retrieved context chunks.
        """
        # If API is disabled, return context raw chunks as fallback
        if not self.enabled:
            return None
            
        if not context_chunks:
            return "Dokümanlarda bu soruya yönelik doğrudan bir bilgi bulunamadı (Yerel arama sonucu boş)."

        # Format retrieved chunks as structured context
        context_str = ""
        for i, item in enumerate(context_chunks):
            chunk = item['chunk']
            score = item['score']
            context_str += f"--- [Paragraf {i+1}] (Kaynak: {chunk['source']}, Sayfa: {chunk['page']}, Alaka Skoru: {score:.2f}) ---\n"
            context_str += f"{chunk['text']}\n\n"

        # System instructions and prompt formatting
        system_instruction = (
            "Sen yardımcı ve dürüst bir yapay zeka asistanısın. Aşağıda verilen referans doküman parçalarına (Context) "
            "bağlı kalarak kullanıcının sorusunu yanıtla. Eğer sorunun cevabı referans dokümanlarda doğrudan yer almıyorsa "
            "veya çıkarılamıyorsa, kesinlikle dışarıdan bilgi uydurma (halüsinasyon görme) ve net bir şekilde "
            "'Verilen dokümanlarda bu bilgiye ulaşılamamıştır.' de.\n"
            "Cevap verirken hangi kaynak dosya ve sayfa numarasından faydalandığını belirt."
        )

        prompt = f"{system_instruction}\n\n[Referans Dokümanlar (Context)]:\n{context_str}\n\n[Kullanıcı Sorusu]:\n{query}\n\n[Cevap]:"

        # Gemini API call using requests
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={self.api_key}"
        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }]
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=20)
            if response.status_code == 200:
                data = response.json()
                # Parse response
                answer = data['candidates'][0]['content']['parts'][0]['text']
                return answer.strip()
            else:
                print(f"[ERROR] Gemini API Hatası. Durum Kodu: {response.status_code}, Yanıt: {response.text}")
                return None
        except Exception as e:
            print(f"[ERROR] Gemini API bağlantı hatası: {e}")
            return None
