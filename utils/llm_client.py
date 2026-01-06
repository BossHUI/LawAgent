import os
from typing import List, Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class LLMClient:
    """
    大语言模型客户端
    @property {str} api_key - API密钥
    @property {str} base_url - API基础地址
    @property {str} model - 模型名
    """
    
    def __init__(self):
        """
        初始化LLM客户端
        @raises ValueError: 如果API密钥不存在
        """
        self.api_key = os.getenv('DEEPSEEK_API_KEY')
        self.base_url = os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')
        self.model = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')
        
        if not self.api_key:
            raise ValueError("请在.env文件中设置DEEPSEEK_API_KEY")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        # 单次请求超时（秒），可通过环境变量覆盖
        try:
            self.request_timeout = float(os.getenv('LLM_REQUEST_TIMEOUT', '60'))
        except Exception:
            self.request_timeout = 60.0
    
    def chat_completion(self, messages: List[Dict], temperature: float = 0.3, max_tokens: Optional[int] = None) -> str:
        """
        聊天补全
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            
        Returns:
            回复内容
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
                timeout=self.request_timeout
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM调用出错：{e}")
            return "抱歉，模型调用出现错误，请稍后重试。"
    
    def generate_with_system_prompt(self, system_prompt: str, user_input: str, temperature: float = 0.7) -> str:
        """
        使用系统提示生成回复
        
        Args:
            system_prompt: 系统提示
            user_input: 用户输入
            temperature: 温度参数
            
        Returns:
            生成的回复
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
        return self.chat_completion(messages, temperature)
    
    def generate_legal_advice(self, question: str, context: str = "", evidence: List[Dict] = None) -> str:
        """
        生成法律建议
        
        Args:
            question: 法律问题
            context: 上下文（对话历史等）
            evidence: 证据列表（从RAG检索得到）
            
        Returns:
            法律建议
        """
        # Force English only
        system_prompt = """You are a professional Singapore legal advisor. Based on the conversation context and automatically retrieved materials (if any), provide accurate and practical legal advice to users.

Requirements:
1. Do not fabricate legal provisions or precedents; if there is insufficient basis, state it clearly.
2. Only refer to "automatically retrieved materials" as "retrieved evidence/reference materials", do not use terms like "user-provided evidence".
3. If the retrieved results do not match the user's question topic, first point out "irrelevant/insufficient relevance" and reduce confidence.
4. Provide clear analysis and actionable recommendations in the main text; cite source titles where appropriate.
5. Maintain professionalism, objectivity, and conciseness.
6. Respond strictly in English. Do not include any Chinese unless it is part of quoted source titles."""

        user_prompt = f"Question: {question}\n\n"
        
        if context:
            user_prompt += f"Context (conversation history):\n{context}\n\n"
        
        if evidence:
            user_prompt += "References retrieved by the system:\n"
            for i, ev in enumerate(evidence[:10], 1):  # Limit to 10 pieces of evidence
                content = ev.get('content', ev.get('text', ''))
                meta = ev.get('metadata', {}) or {}
                source = ev.get('source', meta.get('title', 'Unknown source'))
                user_prompt += f"{i}. [Source: {source}]\n{content[:500]}...\n\n"
        
        return self.generate_with_system_prompt(system_prompt, user_prompt, temperature=0.3)
    
    def generate_contract_review(self, contract_content: str, contract_type: str = "general") -> str:
        """
        生成合同审查报告
        
        Args:
            contract_content: 合同内容
            contract_type: 合同类型
            
        Returns:
            审查报告
        """
        system_prompt = f"""
        You are a professional contract review lawyer, specializing in {contract_type} contracts.

        Please review the following contract and provide a detailed review report, including:
        1. Risk point identification
        2. Missing clause analysis
        3. Suggested modifications
        4. Risk level assessment

        Requirements:
        - Provide specific modification suggestions
        - Mark high-risk clauses
        - Maintain professionalism and objectivity"""

        user_prompt = f"Please review the following {contract_type} contract:\n\n{contract_content}"
        
        return self.generate_with_system_prompt(system_prompt, user_prompt, temperature=0.2)
    
    def generate_contract_draft(self, contract_spec: Dict, contract_type: str = "service") -> str:
        """
        生成合同草稿
        
        Args:
            contract_spec: 合同规格（字典，可能包含检索到的模板和法律条文）
            contract_type: 合同类型
            
        Returns:
            合同草稿
        """
        system_prompt = f"""
        You are a professional contract drafting lawyer, specializing in {contract_type} contracts.

        Please draft a complete contract based on the provided specifications, including:
        1. Standard clause structure
        2. Both parties' rights and obligations
        3. Dispute resolution mechanism
        4. Governing law clause
        Requirements:
        - Clear structure, complete clauses
        - Complies with Singapore legal requirements
        - Include necessary legal protection clauses
        - If provided template reference, can refer to its structure but adjust according to specific specifications
        - If relevant legal provisions, ensure contract complies with legal requirements"""

        # 构建用户提示，包含规格和可能的参考信息
        user_prompt = f"Please draft a {contract_type} contract based on the following specifications:\n\n"
        
        # 提取主要规格信息（排除内部使用的字段）
        main_spec = {k: v for k, v in contract_spec.items() 
                    if not k.startswith('_')}
        
        user_prompt += f"Specifications:\n{main_spec}\n\n"
        
        # 如果有检索到的模板，添加到提示中
        if '_template_reference' in contract_spec:
            template = contract_spec['_template_reference']
            if template:
                user_prompt += f"\nTemplate reference (only for structure reference):\n{template[:1000]}...\n\n"
        
        # 如果有检索到的法律条文，添加到提示中
        if '_retrieved_laws' in contract_spec:
            laws = contract_spec['_retrieved_laws']
            if laws:
                user_prompt += "\nRelevant legal provisions (ensure contract complies with the following requirements):\n"
                for i, law in enumerate(laws[:10], 1):  # 最多10条
                    user_prompt += f"{i}. {law[:500]}...\n"
        
        return self.generate_with_system_prompt(system_prompt, user_prompt, temperature=0.3)