"""
Vector Store - FAISS向量数据库
"""
import os
import pickle
import numpy as np
from typing import List, Dict, Optional

import faiss


class VectorStore:
    """基于FAISS的向量数据库"""
    
    def __init__(self, dimension: int = 768, index_path: Optional[str] = None):
        """
        初始化向量数据库
        
        Args:
            dimension: 向量维度（如果索引已存在，将从索引中读取实际维度）
            index_path: 索引保存路径
        """
        self.index_path = index_path or 'data/vectors/faiss.index'
        # 基于实际使用的 index_path 计算 metadata 和 config 路径，避免当传入 index_path 为 None 时使用错误的默认文件名
        self.metadata_path = self.index_path.replace('.index', '_metadata.pkl')
        self.config_path = self.index_path.replace('.index', '_config.pkl')
        
        # 元数据存储
        self.metadata = []
        # 索引配置信息（模型名称、维度等）
        self.config = {}
        
        # 确保目录存在
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        # 如果索引已存在，加载它（会从索引中获取实际维度）
        if os.path.exists(self.index_path):
            self.load()
            # 从已加载的索引中获取实际维度
            self.dimension = self.index.d
        else:
            # 创建新的FAISS索引
            self.dimension = dimension
            self.index = faiss.IndexFlatL2(dimension)
    
    def add(self, vectors: np.ndarray, metadata: List[Dict]):
        """
        添加向量和元数据
        
        Args:
            vectors: 向量数组
            metadata: 元数据列表
        """
        # 检查维度
        assert vectors.shape[1] == self.dimension, f"向量维度不匹配：{vectors.shape[1]} != {self.dimension}"
        
        # 添加到索引
        if isinstance(self.index, faiss.IndexFlatL2):
            self.index.add(vectors.astype('float32'))
        else:
            # 如果使用其他索引类型，可能需要训练
            self.index.add(vectors.astype('float32'))
        
        # 添加元数据
        self.metadata.extend(metadata)
        
        print(f"已添加 {len(vectors)} 个向量，总计：{self.index.ntotal}")
    
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        搜索相似向量
        
        Args:
            query_vector: 查询向量
            top_k: 返回最相似的k个结果
            
        Returns:
            搜索结果列表
        """
        if self.index.ntotal == 0:
            return []
        
        # 确保是二维数组
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # 检查维度匹配
        query_dim = query_vector.shape[1]
        if query_dim != self.dimension:
            raise ValueError(
                f"查询向量维度 ({query_dim}) 与索引维度 ({self.dimension}) 不匹配。\n"
                f"请确保使用与构建索引时相同的嵌入模型。"
            )
        
        # 转换为float32
        query_vector = query_vector.astype('float32')
        
        # 搜索
        distances, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
        
        # 构建结果
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.metadata):
                meta_item = self.metadata[idx]
                
                # # 调试：检查metadata结构
                # if i < 3:  # 只打印前3个
                #     print(f"[DEBUG VectorStore] 索引 {idx}: meta_item类型={type(meta_item)}, meta_item={meta_item}")
                
                # # 确保meta_item是字典类型
                # if not isinstance(meta_item, dict):
                #     print(f"[DEBUG VectorStore] 警告: 索引 {idx} 的metadata不是字典类型: {type(meta_item)}")
                #     meta_item = {}
                
                # 提取metadata（可能是嵌套的）
                if 'metadata' in meta_item:
                    metadata = meta_item.get('metadata', {})
                else:
                    # 如果metadata不在顶层，可能整个meta_item就是metadata
                    metadata = meta_item.copy()
                    # 移除content和source，它们应该在顶层
                    metadata.pop('content', None)
                    metadata.pop('source', None)
                
                # 确保metadata是字典类型
                if not isinstance(metadata, dict):
                    metadata = {}
                
                source = meta_item.get('source', '') or 'legal'
                content = meta_item.get('content', '')
                
                result = {
                    'content': content,
                    'source': source,
                    'metadata': metadata,
                    'distance': float(distance),
                    'score': 1 / (1 + distance)  # 转换为相似度得分
                }
                
                # # 调试：检查提取的metadata
                # if i < 3:
                #     print(f"[DEBUG VectorStore] 结果 {i}: source={source}, metadata keys={list(metadata.keys())}, metadata={metadata}")
                
                results.append(result)
        
        return results
    
    def save(self, embedder_model_name: Optional[str] = None):
        """
        保存索引和元数据
        
        Args:
            embedder_model_name: 嵌入模型名称（用于后续验证）
        """
        # 保存FAISS索引
        faiss.write_index(self.index, self.index_path)
        
        # 保存元数据
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        # 保存配置信息（模型名称、维度等）
        self.config = {
            'dimension': self.dimension,
            'embedder_model_name': embedder_model_name,
            'total_vectors': self.index.ntotal
        }
        with open(self.config_path, 'wb') as f:
            pickle.dump(self.config, f)
        
        print(f"索引已保存到：{self.index_path}")
        print(f"元数据已保存到：{self.metadata_path}")
        print(f"配置已保存到：{self.config_path}")
    
    def load(self):
        """加载索引和元数据"""
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            
            # # 调试：检查加载的metadata结构
            # if self.metadata:
            #     print(f"[DEBUG VectorStore] 已加载 {len(self.metadata)} 条元数据")
            #     # 检查前3条metadata的结构
            #     for i, meta in enumerate(self.metadata[:3], 1):
            #         print(f"[DEBUG VectorStore] 元数据 {i}: 类型={type(meta)}, keys={list(meta.keys()) if isinstance(meta, dict) else 'N/A'}")
            #         if isinstance(meta, dict):
            #             print(f"[DEBUG VectorStore] 元数据 {i} 内容: {meta}")
        else:
            print(f"[DEBUG VectorStore] 警告: 未找到metadata文件: {self.metadata_path}")
            self.metadata = []
        
        # 加载配置信息
        if os.path.exists(self.config_path):
            with open(self.config_path, 'rb') as f:
                self.config = pickle.load(f)
        else:
            # 向后兼容：如果配置不存在，从索引中获取维度
            self.config = {}
        
        # if not self.metadata:
        #     print(f"[DEBUG VectorStore] 警告: metadata为空，可能需要重新构建索引")
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        stats = {
            'total_vectors': self.index.ntotal,
            'dimension': self.index.d if hasattr(self.index, 'd') else self.dimension,
            'index_type': type(self.index).__name__
        }
        # 添加配置信息
        if self.config:
            stats['embedder_model'] = self.config.get('embedder_model_name', 'unknown')
        return stats

