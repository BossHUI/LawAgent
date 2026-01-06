"""
Document Ingestion Script - 批量处理 data/law 和 data/contract 下文档为向量并写入 data/vectors
支持文件类型：PDF、DOCX
"""
import os
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np

from rag import VectorStore, LegalBERTEmbedder, BM25Retriever
import pickle


def read_pdf_text(pdf_path: Path) -> str:
    """读取 PDF 文本。

    Args:
        pdf_path: PDF 文件路径

    Returns:
        提取的纯文本
    """
    try:
        from pypdf import PdfReader  # 轻量且兼容性好
    except Exception:
        # 兜底
        from PyPDF2 import PdfReader

    reader = PdfReader(str(pdf_path))
    texts: List[str] = []
    for page in getattr(reader, "pages", []):
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        texts.append(page_text)
    return "\n".join(texts)


def read_docx_text(docx_path: Path) -> str:
    """读取 DOCX 文本。"""
    from docx import Document  # python-docx
    try:
        document = Document(str(docx_path))
        paras = [p.text or "" for p in document.paragraphs]
        return "\n".join(paras)
    except Exception:
        return ""


def read_file_text(file_path: Path) -> str:
    """根据后缀调度读取文本。"""
    suffix = (file_path.suffix or "").lower()
    if suffix == ".pdf":
        return read_pdf_text(file_path)
    if suffix == ".docx":
        return read_docx_text(file_path)
    return ""


def chunk_text(text: str, chunk_size: int = 512, chunk_overlap: int = 50, strategy: str = 'semantic') -> List[str]:
    """使用高级语义感知分块对文本进行分块。

    参数：
        text：原始文本
        chunk_size：目标块大小（以令牌为单位）（默认：512，针对 LegalBERT 进行了优化）
        chunk_overlap：tokens重叠（默认：50）
        strategy：分块策略（"语义"、"句子"、"段落"、"固定"）

    返回：
        文本块列表
    """
    # 确保text不为None，并转换为字符串
    if text is None:
        return []
    text = str(text).strip()
    if not text:
        return []
    
    try:
        # 使用高级语义块（如果可用）
        from rag.chunking import chunk_text as semantic_chunk
        return semantic_chunk(text, chunk_size, chunk_overlap, strategy)
    except (ImportError, AttributeError, TypeError) as e:
        # 回退到简单的固定大小分块（如果导入失败或出现其他错误）
        # text 已经在上面的检查中确保不为 None
        
        # 将基于令牌转换为基于字符以进行回退
        char_size = int(chunk_size * 0.75)  # Approximate: 1 token ≈ 0.75 chars
        char_overlap = int(chunk_overlap * 0.75)
        
        chunks: List[str] = []
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


def build_metadata(chunks: List[str], file_path: Path, category: str, source: str = "legal") -> List[Dict]:
    """为每个文本块构建元数据。

    Args:
        chunks: 文本块列表
        file_path: 源文件路径
        category: 类别名称
        source: 数据来源类型 ('legal' 或 'template')

    Returns:
        元数据列表（与 chunks 对齐）
    """
    title = file_path.stem
    rel_dir = file_path.parent
    ftype = (file_path.suffix or "").lstrip(".").lower() or "unknown"
    return [
        {
            "content": chunk,
            "source": source,
            "metadata": {
                "title": title,
                "file": str(file_path),
                "dir": str(rel_dir),
                "category": category,
                "type": ftype
            },
        }
        for chunk in chunks
    ]


def ingest_directory(
    source_dir: Path,
    vector_store: VectorStore,
    embedder: LegalBERTEmbedder,
    source: str = "legal",
    batch_size: int = 64
) -> Tuple[int, List[str]]:
    """处理单个目录下的 PDF 文件并添加到向量库。

    Args:
        source_dir: 源目录路径
        vector_store: 向量库实例
        embedder: 嵌入模型实例
        source: 数据来源类型 ('legal' 或 'template')
        batch_size: 批处理大小

    Returns:
        本次添加的向量数量、收集的chunks列表和metadata列表
    """
    if not source_dir.exists():
        print(f"⚠ 未找到目录：{source_dir}")
        return 0, [], []

    # 递归遍历所有子文件夹中的目标文档类型
    target_files: List[Path] = []
    for pattern in ("*.pdf", "*.docx"):
        target_files.extend(source_dir.rglob(pattern))
    if not target_files:
        print(f"⚠ 目录中未找到可处理的文档（pdf/docx）：{source_dir}")
        return 0, [], []

    print(f"\n开始处理目录：{source_dir} (来源: {source})")
    print(f"找到 {len(target_files)} 个文档")

    total_added = 0
    batch_texts: List[str] = []
    batch_meta: List[Dict] = []
    
    # 收集所有chunks和metadata用于BM25索引
    all_chunks: List[str] = []
    all_metadata: List[Dict] = []

    for file_path in target_files:
        print(f"  读取：{file_path}")
        try:
            text = read_file_text(file_path)
            # 确保文本不为None，并转换为字符串
            if text is None:
                print(f"  ⚠ 跳过空文档或无法解析：{file_path}")
                continue
            text = str(text) if text is not None else ""
            if not text.strip():
                print(f"  ⚠ 跳过空文档或无法解析：{file_path}")
                continue
            
            chunks = chunk_text(text)
            if not chunks:
                print(f"  ⚠ 跳过空文档或无法解析：{file_path}")
                continue

            # 类别：取相对于 source_dir 的第一层目录名（若没有则使用根目录名）
            try:
                relative = file_path.relative_to(source_dir)
                parts = relative.parts
                category = parts[0] if len(parts) > 1 else source_dir.name
            except Exception:
                category = file_path.parent.name or source_dir.name

            metas = build_metadata(chunks, file_path, category, source=source)

            batch_texts.extend(chunks)
            batch_meta.extend(metas)
            all_chunks.extend(chunks)  # 收集用于BM25索引
            # 提取每个chunk的metadata（从完整的meta字典中提取metadata字段）
            for meta in metas:
                # meta的结构是: {'content': ..., 'source': ..., 'metadata': {...}}
                # 我们需要提取metadata字段
                chunk_metadata = meta.get('metadata', {}) if isinstance(meta, dict) else {}
                all_metadata.append(chunk_metadata)

            # 按批写入
            while len(batch_texts) >= batch_size:
                current_texts = batch_texts[:batch_size]
                current_meta = batch_meta[:batch_size]
                del batch_texts[:batch_size]
                del batch_meta[:batch_size]

                vectors = embedder.encode(current_texts)
                vector_store.add(vectors, current_meta)
                total_added += len(current_texts)
                print(f"  已入库：{total_added} 个向量")

        except Exception as e:
            print(f"  ❌ 处理失败 {file_path}：{e}")
            continue

    # 处理剩余不足一批
    if batch_texts:
        try:
            vectors = embedder.encode(batch_texts)
            vector_store.add(vectors, batch_meta)
            total_added += len(batch_texts)
            print(f"  已入库：{total_added} 个向量（最终批）")
        except Exception as e:
            print(f"  ❌ 处理最终批失败：{e}")

    print(f"✓ 目录 {source_dir} 处理完成，新增 {total_added} 个向量")
    
    # 返回总向量数、收集的chunks和metadata
    return total_added, all_chunks, all_metadata


def ingest_pdfs(
    data_dir: Path = Path("data"),
    legal_dir_name: str = "law",
    template_dir_name: str = "contract",
    vectors_dir: Path = Path("data/vectors")
) -> None:
    """批量处理 data/law 和 data/contract 目录下文档（pdf/docx）并写入向量库。

    Args:
        data_dir: 数据根目录（默认 data）
        legal_dir_name: 法律库子目录名（默认 "law"）
        template_dir_name: 合同模板库子目录名（默认 "contract"）
        vectors_dir: 向量库目录（默认 data/vectors）
    """
    legal_dir = data_dir / legal_dir_name
    template_dir = data_dir / template_dir_name

    # 确保向量目录存在
    vectors_dir.mkdir(parents=True, exist_ok=True)

    # 初始化嵌入与向量库（使用 nlpaueb/legal-bert-base-uncased）
    print("正在初始化嵌入模型和向量库...")
    embedder = LegalBERTEmbedder(model_name='nlpaueb/legal-bert-base-uncased')
    
    # 检查是否已有索引，如果有则加载，否则创建新的
    index_path = vectors_dir / "faiss.index"
    if index_path.exists():
        print(f"检测到已有索引，正在加载：{index_path}")
        vector_store = VectorStore(index_path=str(index_path))
        # 确保维度匹配
        if vector_store.dimension != embedder.get_dimension():
            print(f"⚠ 警告：已有索引维度 ({vector_store.dimension}) 与新模型维度 ({embedder.get_dimension()}) 不匹配")
            print("   将创建新索引（旧索引将被覆盖）")
            vector_store = VectorStore(dimension=embedder.get_dimension(), index_path=str(index_path))
    else:
        vector_store = VectorStore(dimension=embedder.get_dimension(), index_path=str(index_path))

    total_added = 0
    all_chunks_for_bm25: List[str] = []
    all_metadata_for_bm25: List[Dict] = []

    # 处理法律库 (data/law)
    added_legal, legal_chunks, legal_metadata = ingest_directory(
        source_dir=legal_dir,
        vector_store=vector_store,
        embedder=embedder,
        source="legal",
        batch_size=64
    )
    total_added += added_legal
    all_chunks_for_bm25.extend(legal_chunks)
    all_metadata_for_bm25.extend(legal_metadata)

    # 处理模板库 (data/contract)
    added_template, template_chunks, template_metadata = ingest_directory(
        source_dir=template_dir,
        vector_store=vector_store,
        embedder=embedder,
        source="template",
        batch_size=64
    )
    total_added += added_template
    all_chunks_for_bm25.extend(template_chunks)
    all_metadata_for_bm25.extend(template_metadata)
    
    # 构建BM25索引
    bm25_index_path = vectors_dir / "bm25_index.pkl"
    if all_chunks_for_bm25:
        print(f"\n正在构建BM25索引...")
        print(f"  文档数量: {len(all_chunks_for_bm25)}")
        bm25_retriever = BM25Retriever()
        bm25_retriever.add_documents(all_chunks_for_bm25, all_metadata_for_bm25)
        bm25_retriever.build_index()
        
        # 保存BM25索引
        try:
            with open(bm25_index_path, 'wb') as f:
                pickle.dump(bm25_retriever, f)
            print(f"✓ BM25索引已保存到: {bm25_index_path}")
        except Exception as e:
            print(f"⚠ 保存BM25索引失败: {e}")
    else:
        print("⚠ 没有文档可用于构建BM25索引")

    # 保存时记录实际使用的模型名称（可能需要触发模型加载）
    embedder.get_dimension()  # 触发模型加载
    model_name = getattr(embedder, '_actual_model_name', None) or getattr(embedder, 'model_name', None) or 'unknown'
    vector_store.save(embedder_model_name=model_name)
    
    stats = vector_store.get_stats()
    print(f"\n{'='*50}")
    print(f"处理完成！")
    print(f"  - 法律库新增向量：{added_legal} 个")
    print(f"  - 模板库新增向量：{added_template} 个")
    print(f"  - 本次总计新增：{total_added} 个")
    print(f"  - 索引总计向量：{stats['total_vectors']} 个")
    print(f"  - 向量维度：{stats['dimension']}")
    print(f"  - 使用模型：{stats.get('embedder_model', 'unknown')}")
    if all_chunks_for_bm25:
        print(f"  - BM25索引已构建：{len(all_chunks_for_bm25)} 个文档")
    print(f"{'='*50}")


if __name__ == "__main__":
    # 允许通过环境变量自定义目录
    data_root = Path(os.environ.get("LAWAGENT_DATA_DIR", "data"))
    legal_name = os.environ.get("LAWAGENT_LEGAL_DIRNAME", "law")
    template_name = os.environ.get("LAWAGENT_TEMPLATE_DIRNAME", "contract")
    vectors_root = Path(os.environ.get("LAWAGENT_VECTORS_DIR", "data/vectors"))
    ingest_pdfs(
        data_dir=data_root,
        legal_dir_name=legal_name,
        template_dir_name=template_name,
        vectors_dir=vectors_root
    )


