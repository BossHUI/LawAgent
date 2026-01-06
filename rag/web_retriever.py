"""
Web Retriever - 外部Web检索器
使用SerpAPI调用搜索引擎获取外部信息
"""
from typing import List, Dict, Optional
import warnings
import os
import requests

warnings.filterwarnings('ignore')


class WebRetriever:
    """基于SerpAPI的Web检索器"""
    
    def __init__(self, api_key: Optional[str] = None):
        """初始化Web检索器
        
        Args:
            api_key: SerpAPI密钥，若未提供则从环境变量SERPAPI_KEY获取
        """
        self.api_key = api_key or os.getenv("SERPAPI_KEY")
        if not self.api_key:
            warnings.warn("未提供SerpAPI密钥，搜索功能将不可用")
    
    def search(self, query: str, max_results: int = 5, region: str = 'sg') -> List[Dict]:
        """
        搜索Web
        
        Args:
            query: 查询文本
            max_results: 最大结果数
            region: 地区代码（sg表示新加坡）
            
        Returns:
            搜索结果列表
        """
        if not self.api_key:
            return []
        
        try:
            # 构建查询（添加新加坡法律相关限定）
            full_query = f"{query} Singapore law"
            
            # SerpAPI请求参数
            params = {
                "q": full_query,
                "api_key": self.api_key,
                "num": max_results,
                "gl": region,  # 地区代码（sg=新加坡）
                "engine": "google"  # 使用Google引擎（可选bing等）
            }
            
            # 调用SerpAPI
            # 设置网络超时，防止长时间阻塞
            response = requests.get("https://serpapi.com/search", params=params, timeout=30)
            response.raise_for_status()  # 抛出HTTP错误
            data = response.json()
            
            # 解析结果
            results = []
            organic_results = data.get("organic_results", [])
            
            for result in organic_results[:max_results]:
                results.append({
                    'content': result.get('snippet', ''),  # 结果摘要
                    'title': result.get('title', ''),
                    'url': result.get('link', ''),
                    'source': 'serpapi',
                    'metadata': {
                        'url': result.get('link', ''),
                        'title': result.get('title', ''),
                        'region': region,
                        'engine': params['engine']
                    }
                })
            
            return results
        
        except requests.Timeout:
            print("Web搜索超时：SerpAPI 请求超过 30s")
            return []
        except Exception as e:
            print(f"Web搜索出错：{e}")
            return []
    
    def assess_confidence(self, result: Dict) -> float:
        """
        评估结果置信度（保持原逻辑）
        
        Args:
            result: 搜索结果
            
        Returns:
            置信度得分（0-1）
        """
        confidence = 0.5  # 基础置信度
        
        # 检查来源域名
        url = result.get('url', '').lower()
        
        # 权威来源加分
        authoritative_domains = [
            'gov.sg',  # 新加坡政府
            'ecitizen.gov.sg',  # 电子公民
            'singaporelawwatch.sg',  # 法律观察
            'singaporelegaladvice.com',  # 法律建议
            'ministryofmannpower.sg',  # 人力部
            'lawgazette.com.sg'  # 法律公报
        ]
        
        for domain in authoritative_domains:
            if domain in url:
                confidence += 0.2
                break
        
        # 限制置信度范围
        return min(confidence, 1.0)