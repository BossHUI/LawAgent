
from typing import List, Dict
from pathlib import Path


class DataLoader:
    """
    数据加载器（兼容性类）
    
    说明：
    - 所有数据都从PDF文件通过 ingest_pdfs.py 导入向量库
    - 数据存储在 data/vectors/faiss.index 中
    - 此类仅用于保持与现有代码的兼容性，返回空列表
    - 实际数据访问通过 RAGRetriever 从向量库检索
    """
    
    def __init__(self, data_dir: str = 'data'):
        """
        初始化数据加载器
        
        Args:
            data_dir: 数据目录（保留参数以保持兼容性，但不使用）
        """
        self.data_dir = Path(data_dir)
        self.data = {
            'legal': [],      # 法律库（空，数据在向量库中）
            'case': [],       # 案例库（空，数据在向量库中）
            'template': []    # 合同模板库（空，数据在向量库中）
        }
    
    def load_all(self):
        """
        加载所有数据（空操作）
        
        说明：
        - PDF数据已通过 ingest_pdfs.py 直接导入向量库
        - 此方法仅用于保持接口兼容性
        """
        print("提示：数据已通过 PDF 文件导入向量库（data/vectors），无需从JSON文件加载")
        print("      请确保已运行: python scripts/ingest_pdfs.py")
    
    def load_legal_data(self):
        """
        加载法律库（空操作）
        
        说明：
        - 法律库数据存储在向量库中（从 data/law/*.pdf 导入）
        - 此方法仅用于保持接口兼容性
        """
        pass
    
    def load_case_data(self):
        """
        加载案例库（空操作）
        
        说明：
        - 案例库数据存储在向量库中（从 data/case/*.pdf 导入，如果有）
        - 此方法仅用于保持接口兼容性
        """
        pass
    
    def load_template_data(self):
        """
        加载合同模板库（空操作）
        
        说明：
        - 模板库数据存储在向量库中（从 data/contract/*.pdf 导入）
        - 此方法仅用于保持接口兼容性
        """
        pass
    
    def get_legal_data(self) -> List[Dict]:
        """
        获取法律库数据
        
        Returns:
            空列表（数据在向量库中）
            
        说明：
        - 实际数据通过 RAGRetriever 从向量库检索
        - 返回空列表以保持接口兼容性
        """
        return self.data['legal']
    
    def get_case_data(self) -> List[Dict]:
        """
        获取案例库数据
        
        Returns:
            空列表（数据在向量库中）
            
        说明：
        - 实际数据通过 RAGRetriever 从向量库检索
        - 返回空列表以保持接口兼容性
        """
        return self.data['case']
    
    def get_template_data(self) -> List[Dict]:
        """
        获取合同模板库数据
        
        Returns:
            空列表（数据在向量库中）
            
        说明：
        - 实际数据通过 RAGRetriever 从向量库检索
        - 返回空列表以保持接口兼容性
        """
        return self.data['template']
    
    def save_data(self, data_type: str, data: List[Dict], filename: str):
        """
        保存数据（已废弃）
        
        说明：
        - 此方法已废弃，因为所有数据都通过 ingest_pdfs.py 导入向量库
        - 保留此方法仅用于保持接口兼容性
        
        Args:
            data_type: 数据类型 ('legal', 'case', 'template')
            data: 数据列表
            filename: 文件名
        """
        print(f"警告：save_data 方法已废弃。数据应通过 scripts/ingest_pdfs.py 导入向量库")

