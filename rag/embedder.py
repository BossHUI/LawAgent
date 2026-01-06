"""
LegalBERT Embedder - 法律文档嵌入模型
"""
import numpy as np
from typing import List, Optional
import warnings
import os
import traceback

# 抑制sentence-transformers的警告
warnings.filterwarnings('ignore')

# 设置 Hugging Face 镜像（如果环境变量未设置）
if 'HF_ENDPOINT' not in os.environ:
    # 可以使用国内镜像站点，如果有网络问题可以取消下面的注释
    # os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # Hugging Face 镜像站点
    pass


class LegalBERTEmbedder:
    """LegalBERT嵌入模型"""
    
    # 默认模型
    DEFAULT_MODEL = 'nlpaueb/legal-bert-base-uncased'
    
    # 备选模型列表（按优先级排序）
    LEGAL_MODEL_ALTERNATIVES = [
        'nlpaueb/legal-bert-base-uncased',  # 标准的 LegalBERT base 模型（默认）
        'nlpaueb/legal-bert-small-uncased',  # LegalBERT small 版本（如果存在）
        'nlpaueb/bert-base-uncased-legal',  # 另一个可能的名称
    ]
    
    # 通用备选模型
    FALLBACK_MODELS = [
        'paraphrase-multilingual-MiniLM-L12-v2',  # 多语言模型
        'all-MiniLM-L6-v2',  # 轻量级通用模型
        'sentence-transformers/all-mpnet-base-v2',  # 高性能通用模型
    ]
    
    def __init__(self, model_name: str = None):
        """
        初始化LegalBERT嵌入模型
        
        Args:
            model_name: 模型名称，如果为None则使用默认模型 nlpaueb/legal-bert-base-uncased
        """
        # 如果没有指定模型，使用默认模型
        self.model_name = model_name if model_name is not None else self.DEFAULT_MODEL
        self.model = None
        self.tokenizer = None
        self._actual_model_name = None  # 实际加载的模型名称
        
    def _setup_huggingface_mirror(self):
        """设置 Hugging Face 镜像站点（如果需要）"""
        # 检查是否已经设置了镜像
        if 'HF_ENDPOINT' in os.environ:
            print(f"使用 Hugging Face 镜像: {os.environ['HF_ENDPOINT']}")
            return
        
        # 检查网络连接问题，提示用户可以使用镜像
        print("提示: 如果遇到网络问题，可以设置环境变量使用镜像站点:")
        print("  Windows: set HF_ENDPOINT=https://hf-mirror.com")
        print("  Linux/Mac: export HF_ENDPOINT=https://hf-mirror.com")
    
    def _load_model(self):
        """加载模型，支持多个备选方案"""
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            
            # 设置镜像（如果需要）
            self._setup_huggingface_mirror()
            
            # 首先尝试用户指定的模型
            models_to_try = []
            if self.model_name:
                models_to_try.append(self.model_name)
            
            # 添加所有法律相关的备选模型
            models_to_try.extend(self.LEGAL_MODEL_ALTERNATIVES)
            
            # 添加通用备选模型
            models_to_try.extend(self.FALLBACK_MODELS)
            
            # 去除重复
            models_to_try = list(dict.fromkeys(models_to_try))
            
            last_error = None
            for model_name in models_to_try:
                try:
                    print(f"正在尝试加载模型: {model_name}")
                    self.model = SentenceTransformer(model_name)
                    self._actual_model_name = model_name
                    
                    if model_name in self.LEGAL_MODEL_ALTERNATIVES:
                        print(f"✓ 成功加载法律模型: {model_name}")
                    elif model_name in self.FALLBACK_MODELS:
                        print(f"⚠ 已回退到通用模型: {model_name}")
                        print(f"   原因: 无法加载指定的法律模型 '{self.model_name}'")
                    else:
                        print(f"✓ 成功加载指定模型: {model_name}")
                    
                    return  # 成功加载，退出
                    
                except Exception as e:
                    last_error = e
                    error_msg = str(e)
                    
                    # 检查是否是网络问题
                    if '404' in error_msg or 'not found' in error_msg.lower():
                        print(f"✗ 模型不存在或无法访问: {model_name}")
                    elif 'timeout' in error_msg.lower() or 'connection' in error_msg.lower():
                        print(f"✗ 网络连接问题，无法加载: {model_name}")
                        print(f"   提示: 请检查网络连接或使用代理访问 Hugging Face")
                    else:
                        print(f"✗ 加载失败: {model_name} - {error_msg[:100]}")
                    continue
            
            # 如果所有模型都加载失败，抛出详细错误
            if self.model is None:
                error_detail = traceback.format_exc() if last_error else "未知错误"
                raise RuntimeError(
                    f"无法加载任何嵌入模型。\n"
                    f"已尝试的模型: {', '.join(models_to_try[:3])}...\n"
                    f"最后错误: {last_error}\n"
                    f"详细错误: {error_detail[:500]}\n"
                    f"请检查:\n"
                    f"1. 网络连接是否正常，能否访问 Hugging Face\n"
                    f"2. 是否安装了必要的依赖: pip install sentence-transformers\n"
                    f"3. 是否可以使用代理或镜像站点"
                )
    
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        编码文本为向量
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            
        Returns:
            向量数组
        """
        self._load_model()
        
        # 如果是单个文本，转换为列表
        if isinstance(texts, str):
            texts = [texts]
        
        # 批量编码
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def encode_single(self, text: str) -> np.ndarray:
        """编码单个文本"""
        return self.encode([text])[0]
    
    def get_dimension(self) -> int:
        """获取向量维度"""
        self._load_model()
        # LegalBERT通常是768维
        return self.model.get_sentence_embedding_dimension()

