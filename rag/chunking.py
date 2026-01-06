"""
高级文本分块策略
支持带有句子/段落边界的语义感知分块
"""
import re
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class ChunkingConfig:
    """分块配置"""
    chunk_size: int = 512  # Target chunk size in tokens (not characters)
    chunk_overlap: int = 50  # Overlap in tokens
    strategy: str = 'semantic'  # 'semantic', 'sentence', 'paragraph', 'fixed'
    min_chunk_size: int = 100  # Minimum chunk size in tokens
    max_chunk_size: int = 1024  # Maximum chunk size in tokens
    respect_sentence_boundary: bool = True
    respect_paragraph_boundary: bool = True


class SemanticChunker:
    """语义感知文本分块器，尊重句子和段落边界"""
    
    # 英语每个字符的近似标记数（粗略估计）
    # Legal text typically has 1 token ≈ 0.75 characters
    TOKENS_PER_CHAR = 1.33
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        """
        Initialize chunker
        
        Args:
            config: Chunking configuration
        """
        self.config = config or ChunkingConfig()
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from text"""
        # Rough estimation: 1 token ≈ 0.75 characters for English
        # 为了更准确的估计，可以使用分词器
        return int(len(text) * self.TOKENS_PER_CHAR)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # 简单的句子分割（可使用 NLTK 或 spaCy 进行改进）
        # Pattern: sentence ends with . ! ? followed by space or newline
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def chunk(self, text: str) -> List[str]:
        """
        基于配置的策略对文本进行分块
        
        参数：
            text：输入文本
            
        返回：
            文本块列表
        """
        # 确保text不为None，并转换为字符串
        if text is None:
            return []
        text = str(text).strip()
        if not text:
            return []
        
        if self.config.strategy == 'semantic':
            return self._semantic_chunk(text)
        elif self.config.strategy == 'sentence':
            return self._sentence_chunk(text)
        elif self.config.strategy == 'paragraph':
            return self._paragraph_chunk(text)
        else:  # 'fixed'
            return self._fixed_chunk(text)
    
    def _semantic_chunk(self, text: str) -> List[str]:
        """语义感知分块，尊重句子和段落边界"""
        chunks = []
        
        # First split into paragraphs if respecting paragraph boundaries
        if self.config.respect_paragraph_boundary:
            paragraphs = self._split_into_paragraphs(text)
        else:
            paragraphs = [text]
        
        for paragraph in paragraphs:
            # Estimate paragraph tokens
            para_tokens = self._estimate_tokens(paragraph)
            
            # If paragraph fits in one chunk, use it as-is
            if para_tokens <= self.config.chunk_size:
                chunks.append(paragraph)
                continue
            
            # Otherwise, split by sentences
            if self.config.respect_sentence_boundary:
                sentences = self._split_into_sentences(paragraph)
            else:
                sentences = [paragraph]
            
            current_chunk = []
            current_tokens = 0
            
            for sentence in sentences:
                sentence_tokens = self._estimate_tokens(sentence)
                
                # If single sentence exceeds max, split it further
                if sentence_tokens > self.config.max_chunk_size:
                    # Split long sentence by commas or semicolons
                    parts = re.split(r'[,;]\s+', sentence)
                    for part in parts:
                        part_tokens = self._estimate_tokens(part)
                        if current_tokens + part_tokens > self.config.chunk_size and current_chunk:
                            # Save current chunk
                            chunk_text = ' '.join(current_chunk)
                            if self._estimate_tokens(chunk_text) >= self.config.min_chunk_size:
                                chunks.append(chunk_text)
                            current_chunk = []
                            current_tokens = 0
                        current_chunk.append(part)
                        current_tokens += part_tokens
                else:
                    # Check if adding sentence would exceed chunk size
                    if current_tokens + sentence_tokens > self.config.chunk_size and current_chunk:
                        # Save current chunk
                        chunk_text = ' '.join(current_chunk)
                        if self._estimate_tokens(chunk_text) >= self.config.min_chunk_size:
                            chunks.append(chunk_text)
                        
                        # Start new chunk with overlap
                        if self.config.chunk_overlap > 0 and current_chunk:
                            # Include last few sentences for overlap
                            overlap_tokens = 0
                            overlap_sentences = []
                            for s in reversed(current_chunk):
                                s_tokens = self._estimate_tokens(s)
                                if overlap_tokens + s_tokens <= self.config.chunk_overlap:
                                    overlap_sentences.insert(0, s)
                                    overlap_tokens += s_tokens
                                else:
                                    break
                            current_chunk = overlap_sentences + [sentence]
                            current_tokens = overlap_tokens + sentence_tokens
                        else:
                            current_chunk = [sentence]
                            current_tokens = sentence_tokens
                    else:
                        current_chunk.append(sentence)
                        current_tokens += sentence_tokens
            
            # Add remaining chunk
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                if self._estimate_tokens(chunk_text) >= self.config.min_chunk_size:
                    chunks.append(chunk_text)
        
        return [c for c in chunks if c.strip()]
    
    def _sentence_chunk(self, text: str) -> List[str]:
        """Chunk by sentences"""
        sentences = self._split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self._estimate_tokens(sentence)
            if current_tokens + sentence_tokens > self.config.chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_tokens = 0
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _paragraph_chunk(self, text: str) -> List[str]:
        """Chunk by paragraphs"""
        paragraphs = self._split_into_paragraphs(text)
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for paragraph in paragraphs:
            para_tokens = self._estimate_tokens(paragraph)
            if current_tokens + para_tokens > self.config.chunk_size and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_tokens = 0
            current_chunk.append(paragraph)
            current_tokens += para_tokens
        
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def _fixed_chunk(self, text: str) -> List[str]:
        """固定大小分块（基于字符，向后兼容）"""
        # Convert token-based config to character-based
        char_size = int(self.config.chunk_size / self.TOKENS_PER_CHAR)
        char_overlap = int(self.config.chunk_overlap / self.TOKENS_PER_CHAR)
        
        chunks = []
        start = 0
        length = len(text)
        
        while start < length:
            end = min(start + char_size, length)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == length:
                break
            start = max(end - char_overlap, start + 1)
        
        return chunks


def chunk_text(
    text: str, 
    chunk_size: int = 512, 
    chunk_overlap: int = 50,
    strategy: str = 'semantic'
) -> List[str]:
    """
    分块文本的便利功能
    
    参数：
        text：输入文本
        chunk_size：令牌中的目标块大小
        chunk_overlap：tokens重叠
        strategy：分块策略（“语义”、“句子”、“段落”、“固定”）
        
    返回：
        文本块列表
    """
    config = ChunkingConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strategy=strategy
    )
    chunker = SemanticChunker(config)
    return chunker.chunk(text)

