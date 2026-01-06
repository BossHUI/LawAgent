"""
诊断脚本：检查 LegalBERT 模型加载问题
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_dependencies():
    """检查依赖是否安装"""
    print("=" * 60)
    print("1. 检查依赖包...")
    print("=" * 60)
    
    required_packages = {
        'sentence_transformers': 'sentence-transformers',
        'transformers': 'transformers',
        'torch': 'torch',
        'numpy': 'numpy'
    }
    
    missing = []
    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"✓ {package} 已安装")
        except ImportError:
            print(f"✗ {package} 未安装")
            missing.append(package)
    
    if missing:
        print(f"\n缺少的包: {', '.join(missing)}")
        print(f"请运行: pip install {' '.join(missing)}")
        return False
    
    return True


def check_network():
    """检查网络连接"""
    print("\n" + "=" * 60)
    print("2. 检查网络连接...")
    print("=" * 60)
    
    import urllib.request
    import socket
    
    # 检查是否能访问 Hugging Face
    test_urls = [
        ('https://huggingface.co', 'Hugging Face 主站'),
        ('https://hf-mirror.com', 'Hugging Face 镜像站'),
    ]
    
    for url, name in test_urls:
        try:
            socket.setdefaulttimeout(5)
            urllib.request.urlopen(url)
            print(f"✓ 可以访问 {name}: {url}")
        except Exception as e:
            print(f"✗ 无法访问 {name}: {url}")
            print(f"  错误: {str(e)[:100]}")


def test_model_loading():
    """测试模型加载"""
    print("\n" + "=" * 60)
    print("3. 测试模型加载...")
    print("=" * 60)
    
    from rag.embedder import LegalBERTEmbedder
    
    try:
        print("\n尝试加载 LegalBERT 嵌入器...")
        embedder = LegalBERTEmbedder()
        
        # 尝试获取维度（这会触发模型加载）
        print("正在获取模型维度...")
        dim = embedder.get_dimension()
        
        print(f"✓ 模型加载成功！")
        print(f"  使用的模型: {embedder._actual_model_name}")
        print(f"  向量维度: {dim}")
        
        # 测试编码
        print("\n测试文本编码...")
        test_text = "这是一个测试文本"
        embedding = embedder.encode_single(test_text)
        print(f"✓ 编码成功！")
        print(f"  测试文本: {test_text}")
        print(f"  向量形状: {embedding.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ 模型加载失败")
        print(f"  错误: {str(e)}")
        import traceback
        print("\n详细错误信息:")
        traceback.print_exc()
        return False


def show_solutions():
    """显示解决方案"""
    print("\n" + "=" * 60)
    print("解决方案")
    print("=" * 60)
    print("""
如果模型加载失败，可以尝试以下方法：

1. 使用 Hugging Face 镜像站点（推荐）:
   Windows:
     set HF_ENDPOINT=https://hf-mirror.com
   
   Linux/Mac:
     export HF_ENDPOINT=https://hf-mirror.com

2. 检查网络连接:
   - 确保能够访问 huggingface.co
   - 如果网络受限，使用 VPN 或代理

3. 更新依赖包:
   pip install --upgrade sentence-transformers transformers torch

4. 清除缓存后重试:
   Windows:
     rmdir /s /q %USERPROFILE%\.cache\huggingface
   
   Linux/Mac:
     rm -rf ~/.cache/huggingface

5. 手动下载模型:
   如果自动下载失败，可以手动从 Hugging Face 下载模型文件
   然后使用本地路径加载
    """)


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("LegalBERT 模型加载诊断工具")
    print("=" * 60)
    print()
    
    # 检查依赖
    if not check_dependencies():
        show_solutions()
        return
    
    # 检查网络
    check_network()
    
    # 测试模型加载
    success = test_model_loading()
    
    # 显示解决方案
    if not success:
        show_solutions()
    
    print("\n" + "=" * 60)
    print("诊断完成")
    print("=" * 60)


if __name__ == "__main__":
    main()

