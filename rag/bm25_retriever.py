"""
BM25 检索器 - 使用 BM25 算法进行基于关键字的检索
"""
from typing import List, Dict, Optional
import re
from collections import Counter
import math


class BM25Retriever:
    """BM25 基于关键字的检索器"""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 retriever
        
        Args:
            k1: 词频饱和参数（通常为 1.2-2.0）
            b: 长度归一化参数（通常为 0.75）
        """
        self.k1 = k1
        self.b = b
        self.documents: List[str] = []
        self.metadata_list: List[Dict] = []  # 存储每个文档的metadata
        self.doc_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}
        self.avg_doc_len: float = 0.0
        self._is_built = False
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        # Convert to lowercase and extract words
        words = re.findall(r'\b\w+\b', text.lower())
        return words
    
    def _calculate_idf(self):
        """Calculate Inverse Document Frequency for all terms"""
        num_docs = len(self.documents)
        if num_docs == 0:
            return
        
        # Count document frequency for each term
        df = Counter()
        for doc_freq in self.doc_freqs:
            df.update(doc_freq.keys())
        
        # Calculate IDF
        for term, doc_freq in df.items():
            # IDF = log((N - df + 0.5) / (df + 0.5))
            # Adding 0.5 to avoid division by zero
            self.idf[term] = math.log((num_docs - doc_freq + 0.5) / (doc_freq + 0.5))
    
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        """
        将文档添加到索引
        
        参数：
            文档：文档文本列表
            metadata：每个文档的可选元数据
        """
        self.documents.extend(documents)
        
        # 存储metadata
        if metadata:
            self.metadata_list.extend(metadata)
        else:
            # 如果没有提供metadata，为每个文档创建空字典
            self.metadata_list.extend([{}] * len(documents))
        
        # Calculate term frequencies for each document
        for doc in documents:
            words = self._tokenize(doc)
            word_freq = Counter(words)
            self.doc_freqs.append(dict(word_freq))
        
        # Calculate average document length
        total_len = sum(len(self._tokenize(doc)) for doc in documents)
        if documents:
            self.avg_doc_len = total_len / len(documents)
        
        self._is_built = False
    
    def build_index(self):
        """Build the BM25 index"""
        if not self.documents:
            return
        
        # Calculate average document length
        if not self.avg_doc_len:
            total_len = sum(len(self._tokenize(doc)) for doc in self.documents)
            self.avg_doc_len = total_len / len(self.documents) if self.documents else 0.0
        
        # Calculate IDF
        self._calculate_idf()
        self._is_built = True
    
    def search(self, query: str, top_k: int = 10, metadata: Optional[List[Dict]] = None) -> List[Dict]:
        """
        使用 BM25 搜索文档
        
        参数：
            query：查询文本
            top_k：要返回的结果数
            metadata：可选元数据列表（与文档顺序相同）
            
        返回：
            BM25 分数的搜索结果列表
        """
        if not self._is_built:
            self.build_index()
        
        if not self.documents:
            return []
        
        query_terms = self._tokenize(query)
        if not query_terms:
            return []
        
        scores = []
        
        for i, doc in enumerate(self.documents):
            doc_freq = self.doc_freqs[i]
            doc_len = len(self._tokenize(doc))
            
            # Calculate BM25 score
            score = 0.0
            for term in query_terms:
                if term in doc_freq:
                    # Term frequency in document
                    tf = doc_freq[term]
                    
                    # IDF
                    idf = self.idf.get(term, 0.0)
                    
                    # BM25 score component
                    # score = IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_doc_len)))
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
                    
                    score += idf * (numerator / denominator)
            
            if score > 0:
                # 优先使用传入的metadata，否则使用存储的metadata
                doc_metadata = {}
                if metadata and i < len(metadata):
                    doc_metadata = metadata[i] if isinstance(metadata[i], dict) else {}
                elif i < len(self.metadata_list):
                    doc_metadata = self.metadata_list[i] if isinstance(self.metadata_list[i], dict) else {}
                
                result = {
                    'content': doc,
                    'score': score,
                    'bm25_score': score,
                    'metadata': doc_metadata,
                    'source': doc_metadata.get('source', 'legal') if doc_metadata else 'legal'
                }
                scores.append((i, result))
        
        # Sort by score (descending)
        scores.sort(key=lambda x: x[1]['score'], reverse=True)
        
        # Return top_k results
        results = [result for _, result in scores[:top_k]]
        return results
    
    def update_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        """Clear and rebuild index with new documents"""
        self.documents = []
        self.doc_freqs = []
        self.idf = {}
        self.avg_doc_len = 0.0
        self.add_documents(documents, metadata)
        self.build_index()

