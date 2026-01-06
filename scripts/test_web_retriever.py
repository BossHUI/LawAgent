"""
Web Retriever 测试脚本
测试联网搜索工具并确保多智能体系统成功使用该工具
"""
import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rag.web_retriever import WebRetriever
from agents.langchain_agents import LegalQAAgent
from utils import ConversationMemory, LLMClient


def test_web_retriever_basic():
    """测试 WebRetriever 基本功能"""
    print("=" * 60)
    print("测试 1: WebRetriever 基本搜索功能")
    print("=" * 60)
    
    # 初始化 WebRetriever
    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        print("⚠️  警告: 未设置 SERPAPI_KEY 环境变量")
        print("   请设置环境变量: export SERPAPI_KEY='your_api_key'")
        print("   或在 Windows 上: set SERPAPI_KEY=your_api_key")
        print("   测试将使用空 API key 进行（会返回空结果）")
    
    web_retriever = WebRetriever(api_key=api_key)
    
    # 测试查询
    test_queries = [
        "employment law Singapore",
        "contract requirements Singapore",
        "company registration Singapore"
    ]
    
    for query in test_queries:
        print(f"\n查询: {query}")
        print("-" * 60)
        
        try:
            results = web_retriever.search(query, max_results=3, region='sg')
            
            if not results:
                print("❌ 未返回结果")
                if not api_key:
                    print("   原因: 未提供 SerpAPI 密钥")
                continue
            
            print(f"✓ 成功检索到 {len(results)} 个结果")
            
            for i, result in enumerate(results, 1):
                print(f"\n结果 {i}:")
                print(f"  标题: {result.get('title', 'N/A')}")
                print(f"  URL: {result.get('url', 'N/A')}")
                print(f"  内容摘要: {result.get('content', 'N/A')[:100]}...")
                print(f"  来源: {result.get('source', 'N/A')}")
                
                # 测试置信度评估
                confidence = web_retriever.assess_confidence(result)
                print(f"  置信度: {confidence:.2f}")
                
        except Exception as e:
            print(f"❌ 搜索出错: {e}")
            import traceback
            traceback.print_exc()


def test_web_retriever_confidence():
    """测试 WebRetriever 置信度评估功能"""
    print("\n" + "=" * 60)
    print("测试 2: WebRetriever 置信度评估")
    print("=" * 60)
    
    web_retriever = WebRetriever()
    
    # 测试不同来源的结果
    test_results = [
        {
            'url': 'https://www.gov.sg/article/employment-act',
            'title': 'Employment Act - Singapore Government',
            'content': 'The Employment Act is the main legislation governing employment in Singapore...',
            'source': 'serpapi'
        },
        {
            'url': 'https://www.singaporelegaladvice.com/employment-law',
            'title': 'Employment Law Guide',
            'content': 'Employment law in Singapore covers various aspects...',
            'source': 'serpapi'
        },
        {
            'url': 'https://www.random-blog.com/employment',
            'title': 'Random Blog Post',
            'content': 'Some general information about employment...',
            'source': 'serpapi'
        }
    ]
    
    for i, result in enumerate(test_results, 1):
        confidence = web_retriever.assess_confidence(result)
        print(f"\n结果 {i}: {result['title']}")
        print(f"  URL: {result['url']}")
        print(f"  置信度: {confidence:.2f}")
        
        if 'gov.sg' in result['url']:
            print("  ✓ 政府网站，置信度应该较高")
        elif 'singaporelegaladvice.com' in result['url']:
            print("  ✓ 法律建议网站，置信度应该较高")
        else:
            print("  ℹ 普通网站，置信度应该较低")


def test_legal_qa_agent_integration():
    """测试与 LegalQAAgent 的集成"""
    print("\n" + "=" * 60)
    print("测试 3: 与 LegalQAAgent 的集成")
    print("=" * 60)
    
    # 初始化组件
    web_retriever = WebRetriever()
    llm_client = LLMClient()
    memory = ConversationMemory(max_history=10)
    memory.new_session()
    
    # 创建 LegalQAAgent（不包含 RAG，仅测试 Web 检索）
    legal_qa_agent = LegalQAAgent(
        rag_retriever=None,  # 不测试 RAG
        web_retriever=web_retriever,
        llm_client=llm_client,
        memory=memory
    )
    
    # 测试问题
    test_questions = [
        "What are the employment law requirements in Singapore?",
        "What is needed to register a company in Singapore?",
        "What are the contract requirements in Singapore?"
    ]
    
    for question in test_questions:
        print(f"\n问题: {question}")
        print("-" * 60)
        
        try:
            session_id = memory.new_session()
            result = legal_qa_agent.answer(question, session_id=session_id)
            
            print(f"✓ 成功生成答案")
            print(f"  置信度: {result.get('confidence', 'N/A')}")
            print(f"  有证据: {result.get('has_evidence', False)}")
            print(f"  主要证据数量: {len(result.get('primary_evidence', []))}")
            print(f"  次要证据数量: {len(result.get('secondary_evidence', []))}")
            
            # 显示 Web 检索结果
            if result.get('secondary_evidence'):
                print(f"\n  Web 检索结果:")
                for i, ev in enumerate(result['secondary_evidence'][:3], 1):
                    meta = ev.get('metadata', {})
                    print(f"    {i}. {meta.get('title', 'N/A')}")
                    print(f"       URL: {meta.get('url', 'N/A')}")
                    print(f"       相关性得分: {ev.get('relevance_score', 0):.2f}")
            
            # 显示答案摘要
            answer = result.get('answer', '')
            if answer:
                print(f"\n  答案摘要: {answer[:200]}...")
            
        except Exception as e:
            print(f"❌ 处理问题出错: {e}")
            import traceback
            traceback.print_exc()


def test_error_handling():
    """测试错误处理"""
    print("\n" + "=" * 60)
    print("测试 4: 错误处理")
    print("=" * 60)
    
    # 测试无 API key 的情况
    print("\n测试无 API key 的情况:")
    web_retriever_no_key = WebRetriever(api_key=None)
    results = web_retriever_no_key.search("test query")
    if not results:
        print("✓ 正确处理：无 API key 时返回空列表")
    else:
        print("❌ 错误：无 API key 时应返回空列表")
    
    # 测试无效查询
    print("\n测试无效查询:")
    web_retriever = WebRetriever()
    try:
        results = web_retriever.search("", max_results=0)
        print(f"✓ 空查询处理正常，返回 {len(results)} 个结果")
    except Exception as e:
        print(f"⚠️  空查询抛出异常: {e}")


def main():
    """主测试函数"""
    parser = argparse.ArgumentParser(description="测试 Web Retriever")
    parser.add_argument("--test", type=str, choices=['basic', 'confidence', 'integration', 'error', 'all'],
                       default='all', help="选择要运行的测试")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Web Retriever 测试套件")
    print("=" * 60)
    
    if args.test in ['basic', 'all']:
        test_web_retriever_basic()
    
    if args.test in ['confidence', 'all']:
        test_web_retriever_confidence()
    
    if args.test in ['integration', 'all']:
        test_legal_qa_agent_integration()
    
    if args.test in ['error', 'all']:
        test_error_handling()
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

