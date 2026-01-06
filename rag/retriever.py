"""
RAG 检索器 - 支持向量和 BM25 的多种检索系统
"""
from typing import List, Dict, Optional
import numpy as np


class RAGRetriever:
    """RAG 检索器 - 支持向量、BM25 和混合检索"""
    
    def __init__(self, vector_store, embedder, bm25_retriever=None):
        """
        初始化 RAG 回收器
        
        参数
            vector_store：矢量数据库
            embedder：嵌入模型
            bm25_retriever：用于混合搜索的可选 BM25 复用器
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.bm25_retriever = bm25_retriever
        self.search_history = []
    
    def search(
        self, 
        query: str, 
        top_k: int = 10, 
        methods: List[str] = ['hybrid'],
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3
    ) -> List[Dict]:
        """
        多检索搜索
        
        参数：
            query：查询文本
            top_k：要返回的结果数
            方法：检索方法 ['vector'， 'bm25'， 'hybrid']
            vector_weight：向量搜索的权重（使用混合时）
            bm25_weight：BM25 搜索的权重（使用混合时）
            
        返回：
            搜索结果
        """
        results = []
        
        # Hybrid search
        if 'hybrid' in methods and self.bm25_retriever is not None:
            results = self._hybrid_search(query, top_k, vector_weight, bm25_weight)
        else:
            # Vector search
            if 'vector' in methods:
                vector_results = self._vector_search(query, top_k)
                results.extend(vector_results)
            
            # BM25 search
            if 'bm25' in methods and self.bm25_retriever is not None:
                bm25_results = self.bm25_retriever.search(query, top_k)
                results.extend(bm25_results)
            
            # Deduplicate
            results = self._deduplicate(results)
            
            # Sort by score
            results.sort(key=lambda x: x.get('score', 0), reverse=True)
            
            # Limit results
            results = results[:top_k]
        
        # 确保每个结果都有metadata字段（即使是空字典）
        for result in results:
            if 'metadata' not in result:
                result['metadata'] = {}
            if 'source' not in result or not result['source']:
                result['source'] = 'legal'
        
        # Record history
        self.search_history.append({
            'query': query,
            'results_count': len(results),
            'methods': methods
        })
        
        return results
    
    def _vector_search(self, query: str, top_k: int) -> List[Dict]:
        """Vector search"""
        # Encode query
        query_vector = self.embedder.encode_single(query)
        
        # Search
        results = self.vector_store.search(query_vector, top_k=top_k)
        
        # 确保每个结果都有metadata字段（即使是空字典）
        for result in results:
            if 'metadata' not in result:
                result['metadata'] = {}
            if 'source' not in result or not result['source']:
                result['source'] = 'legal'
        
        return results
    
    def _deduplicate(self, results: List[Dict]) -> List[Dict]:
        """去重结果"""
        seen = set()
        unique_results = []
        
        for result in results:
            content = result.get('content', '')
            # Use first 200 characters as unique identifier
            content_hash = content[:200] if len(content) > 200 else content
            
            if content_hash not in seen:
                seen.add(content_hash)
                unique_results.append(result)
            else:
                # If duplicate found, keep the one with higher score
                existing_idx = next((i for i, r in enumerate(unique_results) 
                                    if (r.get('content', '')[:200] if len(r.get('content', '')) > 200 else r.get('content', '')) == content_hash), None)
                if existing_idx is not None:
                    existing = unique_results[existing_idx]
                    if result.get('score', 0) > existing.get('score', 0):
                        # 保留新结果，但确保metadata不丢失
                        new_result = result.copy()
                        # 如果新结果的metadata更完整，保留新结果的metadata
                        new_meta = new_result.get('metadata', {}) or {}
                        existing_meta = existing.get('metadata', {}) or {}
                        if not new_meta and existing_meta:
                            new_result['metadata'] = existing_meta
                        elif new_meta and not existing_meta:
                            # 新结果有metadata，保留
                            pass
                        elif new_meta and existing_meta:
                            # 两者都有metadata，保留更完整的
                            if len(new_meta) < len(existing_meta):
                                new_result['metadata'] = existing_meta
                        unique_results[existing_idx] = new_result
        
        return unique_results
    
    def _hybrid_search(
        self, 
        query: str, 
        top_k: int,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3
    ) -> List[Dict]:
        """混合检索：结合向量检索和BM25"""
        # 归一化权重
        total_weight = vector_weight + bm25_weight
        if total_weight > 0:
            vector_weight /= total_weight
            bm25_weight /= total_weight
        
        # 向量检索
        query_vector = self.embedder.encode_single(query)
        vector_results = self.vector_store.search(query_vector, top_k=top_k * 2)
        
        # BM25检索
        bm25_results = self.bm25_retriever.search(query, top_k=top_k * 2)
        
        # 合并结果
        combined_results = {}
        
        # 添加向量结果
        for result in vector_results:
            content = result.get('content', '')
            # 使用内容的前200个字符作为唯一标识，避免相同内容覆盖metadata
            content_key = content[:200] if len(content) > 200 else content
            
            if content_key not in combined_results:
                combined_results[content_key] = {
                    'content': content,
                    'source': result.get('source', '') or 'legal',
                    'metadata': result.get('metadata', {}) or {},
                    'vector_score': result.get('score', 0.0),
                    'bm25_score': 0.0,
                    'combined_score': result.get('score', 0.0) * vector_weight
                }
            else:
                existing = combined_results[content_key]
                # 如果新结果的metadata更完整，则更新
                new_meta = result.get('metadata', {}) or {}
                if new_meta and (not existing.get('metadata') or len(new_meta) > len(existing.get('metadata', {}))):
                    existing['metadata'] = new_meta
                if result.get('source') and not existing.get('source'):
                    existing['source'] = result.get('source')
                existing['vector_score'] = max(existing.get('vector_score', 0), result.get('score', 0))
        
        # 添加BM25结果
        for result in bm25_results:
            content = result.get('content', '')
            # 使用内容的前200个字符作为唯一标识
            content_key = content[:200] if len(content) > 200 else content
            bm25_score = result.get('bm25_score', 0.0)
            
            if content_key not in combined_results:
                combined_results[content_key] = {
                    'content': content,
                    'source': result.get('source', '') or 'legal',
                    'metadata': result.get('metadata', {}) or {},
                    'vector_score': 0.0,
                    'bm25_score': bm25_score,
                    'combined_score': bm25_score * bm25_weight
                }
            else:
                existing = combined_results[content_key]
                # 如果新结果的metadata更完整，则更新
                new_meta = result.get('metadata', {}) or {}
                if new_meta and (not existing.get('metadata') or len(new_meta) > len(existing.get('metadata', {}))):
                    existing['metadata'] = new_meta
                if result.get('source') and not existing.get('source'):
                    existing['source'] = result.get('source')
                existing['bm25_score'] = bm25_score
                existing['combined_score'] = (
                    existing.get('vector_score', 0) * vector_weight +
                    bm25_score * bm25_weight
                )
        
        # 归一化分数到[0, 1]范围
        if combined_results:
            max_score = max(r['combined_score'] for r in combined_results.values())
            min_score = min(r['combined_score'] for r in combined_results.values())
            score_range = max_score - min_score if max_score > min_score else 1.0
            
            for result in combined_results.values():
                if score_range > 0:
                    result['combined_score'] = (result['combined_score'] - min_score) / score_range
                result['score'] = result['combined_score']
        
        # 按组合得分排序
        final_results = list(combined_results.values())
        final_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # 确保每个结果都有metadata字段（即使是空字典）
        for result in final_results:
            if 'metadata' not in result:
                result['metadata'] = {}
            if 'source' not in result or not result['source']:
                result['source'] = 'legal'
        
        return final_results[:top_k]
    
    def search_by_source(self, query: str, source: str, top_k: int = 5) -> List[Dict]:
        """
        按来源检索
        
        Args:
            query: 查询文本
            source: 来源类型 ('legal', 'case', 'template')
            top_k: 返回数量
            
        Returns:
            搜索结果
        """
        all_results = self.search(query, top_k=top_k*2)
        
        # 过滤特定来源
        filtered_results = [
            r for r in all_results
            if r.get('source') == source
        ]
        
        return filtered_results[:top_k]
    
    def get_search_statistics(self) -> Dict:
        """获取检索统计信息"""
        return {
            'total_searches': len(self.search_history),
            'average_results': np.mean([s['results_count'] for s in self.search_history]) if self.search_history else 0
        }

