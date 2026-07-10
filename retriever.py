import re
import math
import numpy as np

class DocumentRetriever:
    def __init__(self, chunk_size=500, chunk_overlap=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks = []      # list of dicts: {'text': str, 'source': str, 'page': int}
        self.vocab = {}       # word -> index mapping
        self.idf = {}         # word -> idf value
        self.tfidf_matrix = None # numpy array of shape (num_chunks, vocab_size)

    def _tokenize(self, text):
        """
        Tokenizes text by lowercasing and matching alphanumeric words.
        """
        # Turkish lowercasing adjustments if needed (simple .lower() works for general use)
        words = re.findall(r'\w+', text.lower())
        # Filter out numbers and very short words
        return [w for w in words if len(w) > 1]

    def add_document(self, file_path, pages):
        """
        Splits pages into chunks and adds them to the retriever.
        pages: list of dicts [{'page': int, 'text': str}]
        """
        source_name = file_path.split('/')[-1]
        
        for p in pages:
            text = p['text']
            page_num = p['page']
            
            # Simple text splitter with overlap
            i = 0
            if len(text) <= self.chunk_size:
                self.chunks.append({
                    'text': text,
                    'source': source_name,
                    'page': page_num
                })
            else:
                while i < len(text):
                    chunk_text = text[i:i + self.chunk_size]
                    self.chunks.append({
                        'text': chunk_text,
                        'source': source_name,
                        'page': page_num
                    })
                    i += (self.chunk_size - self.chunk_overlap)

    def build_index(self):
        """
        Builds the TF-IDF index for all added chunks.
        """
        if not self.chunks:
            print("[WARN] İndekslenecek doküman parçası bulunamadı.")
            return

        # 1. Build Vocabulary and Document Frequencies
        doc_frequencies = {}
        vocab_set = set()
        
        # Tokenize each chunk and calculate term occurrences
        chunk_tokens = []
        for chunk in self.chunks:
            tokens = self._tokenize(chunk['text'])
            chunk_tokens.append(tokens)
            vocab_set.update(tokens)
            
            # Count document frequencies (how many chunks contain the word)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                doc_frequencies[token] = doc_frequencies.get(token, 0) + 1

        self.vocab = {word: idx for idx, word in enumerate(sorted(vocab_set))}
        vocab_size = len(self.vocab)
        num_chunks = len(self.chunks)

        if vocab_size == 0:
            print("[WARN] Kelime haznesi boş. İndeks oluşturulamadı.")
            return

        # 2. Compute IDF values
        # idf = ln(1 + (N / DF))
        for word, df in doc_frequencies.items():
            self.idf[word] = math.log(1 + (num_chunks / df))

        # 3. Compute TF-IDF Matrix
        self.tfidf_matrix = np.zeros((num_chunks, vocab_size), dtype=np.float32)
        
        for chunk_idx, tokens in enumerate(chunk_tokens):
            if not tokens:
                continue
            # Calculate Term Frequencies (TF = raw count / total tokens in chunk)
            tf = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            
            for token, count in tf.items():
                if token in self.vocab:
                    word_idx = self.vocab[token]
                    tf_val = count / len(tokens)
                    self.tfidf_matrix[chunk_idx, word_idx] = tf_val * self.idf[token]
                    
        print(f"[INFO] İndeksleme tamamlandı. {num_chunks} paragraf ve {vocab_size} benzersiz kelime indekslendi.")

    def retrieve(self, query, top_k=3):
        """
        Retrieves the top K most relevant chunks for the query using Cosine Similarity.
        """
        if self.tfidf_matrix is None or len(self.chunks) == 0:
            return []

        # 1. Vectorize query
        query_tokens = self._tokenize(query)
        query_vector = np.zeros(len(self.vocab), dtype=np.float32)
        
        if not query_tokens:
            return []

        # Calculate query TF-IDF
        query_tf = {}
        for token in query_tokens:
            query_tf[token] = query_tf.get(token, 0) + 1
            
        for token, count in query_tf.items():
            if token in self.vocab:
                word_idx = self.vocab[token]
                tf_val = count / len(query_tokens)
                query_vector[word_idx] = tf_val * self.idf[token]

        # Calculate norms for Cosine Similarity
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            # If query has no words from vocab, return random/first chunks or empty
            return []

        # Calculate cosine similarity: (A dot B) / (||A|| * ||B||)
        dot_products = np.dot(self.tfidf_matrix, query_vector)
        matrix_norms = np.linalg.norm(self.tfidf_matrix, axis=1)
        
        # Avoid division by zero
        matrix_norms[matrix_norms == 0] = 1e-9
        
        similarities = dot_products / (matrix_norms * query_norm)

        # Get top K indices sorted by similarity score
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            # Only return chunks with similarity > 0 (or some small positive float)
            if score > 0.0:
                results.append({
                    'chunk': self.chunks[idx],
                    'score': score
                })
        return results
