"""
Re-ranker - 重排序器
对检索结果进行重新排序以提高相关性，支持 cross-encoder 模型
"""
from typing import List, Dict, Optional
import re


class Reranker:
    """重排序器 - 支持关键词匹配和 cross-encoder 模型"""
    
    def __init__(self, use_cross_encoder: bool = False, model_name: Optional[str] = None):
        """
        初始化重排序器
        
        Args:
            use_cross_encoder: 是否使用 cross-encoder 模型
            model_name: Cross-encoder 模型名称（默认: 'cross-encoder/ms-marco-MiniLM-L-12-v2'）
        """
        self.use_cross_encoder = use_cross_encoder
        self.model_name = model_name or 'cross-encoder/ms-marco-MiniLM-L-12-v2'
        self.cross_encoder = None
        
        if use_cross_encoder:
            self._load_cross_encoder()
        
        # 相关性关键词
        self.relevance_keywords = {
            'high': ['regulation', 'act', 'law', 'code', 'statute', 'provision', 'clause', 'article'],
            'medium': ['guideline', 'practice', 'policy', 'principle', 'standard'],
            'low': ['may', 'might', 'possible', 'sometimes']
        }
    
    def _load_cross_encoder(self):
        """加载 cross-encoder 模型"""
        try:
            from sentence_transformers import CrossEncoder
            print(f"Loading cross-encoder model: {self.model_name}")
            self.cross_encoder = CrossEncoder(self.model_name)
            print("✓ Cross-encoder model loaded successfully")
        except ImportError:
            print("⚠ sentence-transformers not available for cross-encoder, falling back to keyword-based reranking")
            self.use_cross_encoder = False
        except Exception as e:
            print(f"⚠ Failed to load cross-encoder model: {e}")
            print("   Falling back to keyword-based reranking")
            self.use_cross_encoder = False
    
    def rerank(
        self, 
        query: str, 
        results: List[Dict], 
        top_k: Optional[int] = None,
        use_cross_encoder: Optional[bool] = None
    ) -> List[Dict]:
        """
        重排序结果
        
        Args:
            query: 查询文本
            results: 原始结果列表
            top_k: 返回前k个结果
            use_cross_encoder: 覆盖默认 cross-encoder 设置
            
        Returns:
            重排序后的结果列表
        """
        use_ce = use_cross_encoder if use_cross_encoder is not None else self.use_cross_encoder
        
        if use_ce and self.cross_encoder is not None:
            return self._rerank_with_cross_encoder(query, results, top_k)
        else:
            return self._rerank_with_keywords(query, results, top_k)
    
    def _rerank_with_cross_encoder(self, query: str, results: List[Dict], top_k: Optional[int] = None) -> List[Dict]:
        """使用 cross-encoder 模型重排序"""
        if not results:
            return []
        
        # 准备 cross-encoder 的查询-文档对
        pairs = []
        for result in results:
            content = result.get('content', '')
            # 截断内容（cross-encoder 有最大长度限制）
            max_length = 512
            if len(content) > max_length:
                content = content[:max_length]
            pairs.append([query, content])
        
        # 从 cross-encoder 获取分数
        try:
            scores = self.cross_encoder.predict(pairs)
            
            # 将分数添加到结果中
            reranked = []
            for i, result in enumerate(results):
                result_copy = result.copy()
                result_copy['rerank_score'] = float(scores[i])
                result_copy['cross_encoder_score'] = float(scores[i])
                
                # 结合原始得分
                original_score = result.get('score', 0)
                if original_score > 0:
                    # 加权组合：70% cross-encoder，30% 原始得分
                    result_copy['combined_score'] = 0.7 * float(scores[i]) + 0.3 * original_score
                else:
                    result_copy['combined_score'] = float(scores[i])
                
                reranked.append(result_copy)
            
            # 按组合得分排序
            reranked.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
            
            if top_k:
                reranked = reranked[:top_k]
            
            return reranked
            
        except Exception as e:
            print(f"⚠ Cross-encoder prediction failed: {e}, falling back to keyword-based")
            return self._rerank_with_keywords(query, results, top_k)
    
    def _rerank_with_keywords(self, query: str, results: List[Dict], top_k: Optional[int] = None) -> List[Dict]:
        """使用关键词匹配重排序（后备方法）"""
        reranked = []
        
        for result in results:
            relevance_score = self._calculate_relevance(query, result)
            
            # 结合原始得分和相关性得分
            original_score = result.get('score', 0)
            combined_score = 0.6 * original_score + 0.4 * relevance_score
            
            result_copy = result.copy()
            result_copy['rerank_score'] = combined_score
            result_copy['relevance_score'] = relevance_score
            result_copy['combined_score'] = combined_score
            reranked.append(result_copy)
        
        # 按组合得分排序
        reranked.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
        
        if top_k:
            reranked = reranked[:top_k]
        
        return reranked
    
    def _calculate_relevance(self, query: str, result: Dict) -> float:
        """计算基于关键词的相关性得分"""
        content = result.get('content', '').lower()
        query_lower = query.lower()
        
        score = 0.0
        
        # 提取查询词
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        content_words = set(re.findall(r'\b\w+\b', content))
        
        # 单词重叠率
        if query_words:
            overlap = len(query_words.intersection(content_words)) / len(query_words)
            score += overlap * 0.5
        
        # 检查相关性关键词
        high_count = sum(1 for kw in self.relevance_keywords['high'] if kw in content)
        medium_count = sum(1 for kw in self.relevance_keywords['medium'] if kw in content)
        
        score += min(high_count * 0.1, 0.3)
        score += min(medium_count * 0.05, 0.2)
        
        # 标题提升
        metadata = result.get('metadata', {})
        title = metadata.get('title', '').lower()
        
        if any(word in title for word in query_lower.split()):
            score += 0.1
        
        return min(score, 1.0)
    
    def filter_by_threshold(self, results: List[Dict], threshold: float = 0.3) -> List[Dict]:
        """
        根据阈值过滤结果
        
        Args:
            results: 结果列表
            threshold: 得分阈值
            
        Returns:
            过滤后的结果
        """
        return [r for r in results if r.get('combined_score', r.get('rerank_score', 0)) >= threshold]

