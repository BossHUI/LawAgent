"""
RAG 测试脚本 - 载入 data/vectors 索引，对查询进行检索与可选重排序
"""
import os
import argparse
from pathlib import Path
from typing import List

from rag import VectorStore, LegalBERTEmbedder, RAGRetriever, Reranker


"""

    python scripts/test_rag.py --query "What legal requirements are needed for signing a contract?" --top_k 5
  - 启用重排序测试：
    python scripts/test_rag.py --query "What are the legal circumstances for the termination of a labor contract?" --top_k 8 --rerank

- 输出示例要点：
  - 显示 score（相似度）及可选的 rerank 分数
  - 显示 `metadata.category`（来自子文件夹名）与 `metadata.title`（文件名）
  - 展示内容片段摘要，便于快速验证相关性
"""

# /**
#  * 解析命令行参数
#  * @returns argparse.Namespace
#  */
def parse_args():
    parser = argparse.ArgumentParser(description="测试当前 RAG 系统")
    parser.add_argument("--query", type=str, default="合同签订需要具备哪些法律要件？", help="测试查询问题")
    parser.add_argument("--top_k", type=int, default=5, help="返回结果数量")
    parser.add_argument("--rerank", action="store_true", help="是否启用重排序")
    parser.add_argument("--vectors_dir", type=str, default=os.environ.get("LAWAGENT_VECTORS_DIR", "data/vectors"), help="向量库目录")
    return parser.parse_args()


# /**
#  * 打印检索结果
#  * @param {list} results - 检索结果
#  * @param {bool} show_rerank - 是否展示 rerank 分数
#  */
def print_results(results, show_rerank: bool = False):
    if not results:
        print("未检索到结果。")
        return
    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {}) or {}
        title = meta.get("title", "")
        category = meta.get("category", meta.get("type", ""))
        score = r.get("score", 0.0)
        rerank_score = r.get("rerank_score", None)
        snippet = (r.get("content", "") or "").replace("\n", " ")[:140]
        line = f"[{i}] score={score:.4f}"
        if show_rerank and rerank_score is not None:
            line += f", rerank={rerank_score:.4f}"
        print(line)
        print(f"    类别: {category} | 标题: {title}")
        print(f"    摘要: {snippet}...")


def main():
    args = parse_args()

    vectors_dir = Path(args.vectors_dir)
    index_path = vectors_dir / "faiss.index"

    if not index_path.exists():
        print(f"未找到索引文件：{index_path}")
        print("请先运行向量构建脚本：python scripts/ingest_pdfs.py")
        return

    # 先加载向量库以获取配置信息
    vector_store = VectorStore(dimension=768, index_path=str(index_path))  # 临时维度，会被覆盖
    stats = vector_store.get_stats()
    index_dim = stats['dimension']
    index_model = stats.get('embedder_model', None)
    
    # 根据索引配置选择合适的嵌入模型
    embedder = None
    if index_model and index_model != 'unknown':
        # 尝试使用构建索引时使用的模型
        try:
            embedder = LegalBERTEmbedder(model_name=index_model)
            if embedder.get_dimension() != index_dim:
                print(f"警告：模型 {index_model} 的维度 ({embedder.get_dimension()}) 与索引维度 ({index_dim}) 不匹配")
                embedder = None
        except Exception as e:
            print(f"无法加载模型 {index_model}：{e}")
            embedder = None
    
    # 如果无法使用指定模型，尝试自动匹配
    if embedder is None or embedder.get_dimension() != index_dim:
        # 根据维度自动选择合适的模型
        print(f"索引维度：{index_dim}，尝试自动匹配嵌入模型...")
        
        # 根据维度自动匹配模型（常见维度对应关系）
        if index_dim == 384:
            # 384维通常是 paraphrase-multilingual-MiniLM-L12-v2
            embedder = LegalBERTEmbedder(model_name='paraphrase-multilingual-MiniLM-L12-v2')
        elif index_dim == 768:
            # 768维可能是 LegalBERT 或其他BERT变体
            embedder = LegalBERTEmbedder(model_name='nlpaueb/legal-bert-small-uncased')
        else:
            # 其他维度，尝试默认模型
            embedder = LegalBERTEmbedder()
        
        # 验证维度匹配
        embedder_dim = embedder.get_dimension()
        if embedder_dim != index_dim:
            print(f"错误：无法找到维度为 {index_dim} 的嵌入模型（当前模型维度：{embedder_dim}）")
            print(f"索引统计信息：{stats}")
            print("\n解决方案：")
            print(f"1. 重新构建索引：python scripts/ingest_pdfs.py")
            print("2. 或手动指定正确的模型名称")
            return
    
    embedder_dim = embedder.get_dimension()
    print(f"使用嵌入模型：{getattr(embedder, '_actual_model_name', embedder.model_name)} (维度: {embedder_dim})")
    
    retriever = RAGRetriever(vector_store, embedder)
    reranker = Reranker() if args.rerank else None

    print("==== RAG 检索测试 ====")
    print(f"Query: {args.query}")
    print(f"Top-K: {args.top_k} | Rerank: {bool(reranker)}")

    # 检索
    base_results = retriever.search(args.query, top_k=args.top_k, methods=["vector"]) or []

    # 重排序（可选）
    if reranker and base_results:
        reranked = reranker.rerank(args.query, base_results, top_k=args.top_k)
        print_results(reranked, show_rerank=True)
    else:
        print_results(base_results, show_rerank=False)


if __name__ == "__main__":
    main()


