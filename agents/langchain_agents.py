
from typing import Dict, List, Optional, Any
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os
import sys

from utils.llm_client import LLMClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class LegalQAAgent:
    """
    商业法律问答智能体
    基于RAG、web检索器、外部LLM能力，支持证据归纳的法律答疑服务。
    """
    def __init__(self, rag_retriever=None, web_retriever=None, reranker=None, llm_client=None, memory=None):
        from utils.llm_client import LLMClient
        self.rag_retriever = rag_retriever
        self.web_retriever = web_retriever
        self.reranker = reranker
        self.llm_client = llm_client or LLMClient()
        self.memory = memory
        self.qa_history = []
        
    def answer(self, question: str, context: dict = None, session_id: str = None) -> dict:
        """
        回答问题并返回法律证据与置信度评估
        Args:
            question: 用户问题
            context: 上下文（字典，可选）
            session_id: 会话ID（用于从memory获取上下文）
        Returns:
            答案与证据
        """
        # 从memory获取对话上下文（如果有session_id和memory）
        context_str = ""
        if context and isinstance(context, dict) and 'context' in context:
            context_str = context['context']
        elif self.memory and session_id:
            try:
                context_str = self.memory.get_context(session_id, max_messages=5)
            except Exception as e:
                print(f"获取对话上下文时出错：{e}")
        
        # 判断是否是简单的问候类问题，如果是则跳过RAG检索
        if self._is_simple_greeting(question):
            # 简单问候类问题，直接使用LLM生成答案，不进行检索
            answer = self.llm_client.generate_legal_advice(
                question, context=context_str, evidence=[])
            result = {
                'answer': answer,
                'confidence': 'low',
                'primary_evidence': [],
                'secondary_evidence': [],
                'has_evidence': False,
                'uncertainty_note': ''
            }
            self.qa_history.append({'question': question, 'result': result, 'timestamp': self._get_timestamp()})
            return result
        
        # 使用RAG检索
        rag_results = self._retrieve_from_rag(question)
        # 使用Web检索器检索
        web_results = self._retrieve_from_web(question)
        
        # 调试日志：记录检索结果数量
        print(f"[DEBUG] RAG检索结果数量: {len(rag_results)}")
        print(f"[DEBUG] Web检索结果数量: {len(web_results)}")
        if rag_results:
            print(f"[DEBUG] RAG检索前3个结果来源: {[r.get('source', 'unknown') for r in rag_results[:3]]}")
            print(f"[DEBUG] RAG检索前3个结果得分: {[r.get('score', 0) for r in rag_results[:3]]}")
            # # 调试：检查前3个结果的metadata结构
            # for i, r in enumerate(rag_results[:3], 1):
            #     meta = r.get('metadata', {})
            #     print(f"[DEBUG] RAG结果 {i} - source: {r.get('source', 'N/A')}, metadata keys: {list(meta.keys()) if meta else 'None'}, metadata: {meta}")

        primary_evidence = self._evaluate_evidence(rag_results, 'rag', question)
        secondary_evidence = self._evaluate_evidence(web_results, 'web', question)
        
        # 调试日志：记录评估后的证据数量
        print(f"[DEBUG] 评估后的RAG证据数量: {len(primary_evidence)}")
        print(f"[DEBUG] 评估后的Web证据数量: {len(secondary_evidence)}")
        if primary_evidence:
            print(f"[DEBUG] RAG证据相关性得分: {[e.get('relevance_score', 0) for e in primary_evidence]}")
            # 改进调试日志：显示完整的metadata信息
            for i, ev in enumerate(primary_evidence[:3], 1):
                meta = ev.get('metadata', {}) or {}
                title = meta.get('title', 'N/A')
                file_path = meta.get('file', 'N/A')
                category = meta.get('category', 'N/A')
                # print(f"[DEBUG] RAG证据 {i} - title: {title}, file: {file_path}, category: {category}")
        
        has_evidence = len(primary_evidence) > 0 or len(secondary_evidence) > 0

        # 依据证据相关性确定置信度
        all_evs = primary_evidence + secondary_evidence
        avg_score = sum(e.get('relevance_score', 0) for e in all_evs) / len(all_evs) if all_evs else 0.0

        # 使用llm_client生成答案，并传递上下文（措辞不宣称“用户提供的证据”）
        if len(primary_evidence) > 0:
            answer = self.llm_client.generate_legal_advice(
                question, context=context_str, evidence=all_evs)
            confidence = 'high' if avg_score >= 0.75 else ('medium' if avg_score >= 0.5 else 'low')
        elif len(secondary_evidence) > 0:
            answer = self.llm_client.generate_legal_advice(
                question, context=context_str if context_str else "The following are automatically retrieved external materials, for reference only", 
                evidence=secondary_evidence)
            confidence = 'medium' if avg_score >= 0.5 else 'low'
        else:
            answer = self.llm_client.generate_legal_advice(
                question, context=context_str if context_str else "No highly relevant materials were retrieved for this question", evidence=[])
            confidence = 'low'
        result = {
            'answer': answer,
            'confidence': confidence,
            'primary_evidence': primary_evidence,
            'secondary_evidence': secondary_evidence,
            'has_evidence': has_evidence,
            'uncertainty_note': self._generate_uncertainty_note(confidence) if has_evidence else ''
        }
        self.qa_history.append({'question': question, 'result': result, 'timestamp': self._get_timestamp()})
        return result
    
    def _is_simple_greeting(self, question: str) -> bool:
        """判断是否是简单的问候类问题，不需要检索"""
        question_lower = question.lower().strip()
        
        # 简单的问候语
        greeting_patterns = [
            '你好', 'hello', 'hi', 'hey',
            '你是谁', 'who are you', 'what are you',
            '你是什么', 'what is', 'what\'s',
            '介绍', 'introduce', '介绍自己',
            '谢谢', 'thank you', 'thanks',
            '再见', 'bye', 'goodbye',
            '早上好', 'good morning',
            '下午好', 'good afternoon',
            '晚上好', 'good evening',
            '你好吗', 'how are you',
            '很高兴', 'nice to meet',
            '嗨', 'hey there'
        ]
        
        # 检查是否完全是问候语（不包含法律相关关键词）
        is_greeting_only = any(pattern in question_lower for pattern in greeting_patterns)
        
        # 如果包含法律相关关键词，则不是简单问候
        legal_keywords = [
            '法律', 'law', 'legal', '合同', 'contract', '协议', 'agreement',
            '法规', 'regulation', '条例', 'act', 'statute', '条款', 'clause',
            '案例', 'case', '判决', 'judgment', '争议', 'dispute',
            '咨询', '咨询', 'advice', '需要', 'need', 'require', 'requirements'
        ]
        has_legal_keyword = any(keyword in question_lower for keyword in legal_keywords)
        
        # 如果只是问候语且没有法律关键词，则认为是简单问候
        if is_greeting_only and not has_legal_keyword:
            return True
        
        # 如果问题太短且没有法律关键词，也可能是简单问候
        if len(question.strip()) <= 10 and not has_legal_keyword:
            return True
        
        return False
    def _retrieve_from_rag(self, question: str) -> list:
        if self.rag_retriever:
            # 预处理查询：提取关键信息，去除冗余词汇
            # 保留原始查询，因为向量模型能理解完整语义
            processed_query = question.strip()
            
            # legal_qa只检索法律条文，过滤掉合同模板
            # 使用search_by_source方法，只检索source='legal'的结果
            results = self.rag_retriever.search_by_source(processed_query, source='legal', top_k=10)
            
            # 使用reranker对结果进行重排序，提高相关性
            if self.reranker and results:
                results = self.reranker.rerank(processed_query, results, top_k=10)
            
            return results
        return []
    def _retrieve_from_web(self, question: str) -> list:
        if self.web_retriever:
            results = self.web_retriever.search(question, max_results=3)
            return results
        return []
    def _evaluate_evidence(self, results: list, source_type: str, question: str = "") -> list:
        evidence = []
        q = (question or "").lower()
        
        # 从问题中提取关键实体和主题关键词
        topic_keywords = []
        
        # 劳动相关主题
        if any(k in q for k in ['labor', 'employment', 'employee', 'employer', 'manpower', 'work', 'workplace', 'salary']) or any(k in q for k in ['劳动', '雇佣', '员工', '雇主', '人力', '工资', '工时', '加班']):
            topic_keywords.extend(['labor', 'employment', 'employee', 'employer', 'manpower', 'work', 'workplace', 'salary', '劳动', '雇佣', '员工', '雇主', '人力', '工资', '工时', '加班'])
        
        # 公司与商业主体相关主题
        if any(k in q for k in ['公司', '商业', '企业', '主体', '纠纷', 'company', 'business', 'corporate', 'entity', 'dispute', 'corporation', 'commercial']):
            topic_keywords.extend(['公司', '商业', '企业', '主体', '纠纷', '公司', '法人', 'corporation', 'company', 'business', 'corporate', 'entity', 'dispute', 'commercial', 'enterprise', 'firm', 'partnership', 'limited', 'llc', 'incorporation', 'shareholder', 'director', 'board', '股东', '董事', '董事会', '注册', '设立', '清算', '破产'])
        
        # 合同与商事规则相关主题
        if any(k in q for k in ['合同', '协议', '商事', 'contract', 'agreement', 'commercial', 'business', 'transaction']):
            topic_keywords.extend(['合同', '协议', '商事', 'contract', 'agreement', 'commercial', 'business', 'transaction', '买卖', 'sale', 'purchase', '交易', 'trade'])
        
        # 知识产权相关主题
        if any(k in q for k in ['知识产权', '专利', '商标', '版权', 'intellectual', 'property', 'patent', 'trademark', 'copyright']):
            topic_keywords.extend(['知识产权', '专利', '商标', '版权', 'intellectual', 'property', 'patent', 'trademark', 'copyright', 'ip', 'ipr'])
        
        # 金融与债务相关主题
        if any(k in q for k in ['金融', '债务', '融资', 'finance', 'debt', 'loan', 'financing']):
            topic_keywords.extend(['金融', '债务', '融资', 'finance', 'debt', 'loan', 'financing', 'credit', '银行', 'bank'])
        
        # 仲裁与争议解决相关主题
        if any(k in q for k in ['仲裁', '争议', '解决', 'arbitration', 'dispute', 'resolution']):
            topic_keywords.extend(['仲裁', '争议', '解决', 'arbitration', 'dispute', 'resolution', 'mediation', 'litigation', '诉讼', '调解'])

        for i, result in enumerate(results):
            content_full = result.get('content', result.get('text', ''))
            content = (content_full or '').lower()
            
            # 获取向量相似度得分（如果存在）
            vector_score = result.get('score', 0.0)  # 向量检索的相似度得分 (0-1)
            distance = result.get('distance', float('inf'))
            if distance != float('inf') and distance > 0:
                # 如果距离较小，说明相似度高
                vector_score = 1 / (1 + distance)
            
            # 通用法律/案例信号
            legal_keywords = ['法', '律', 'law', 'legal', 'regulation', 'act', 'statute', 'legislation', 'code']
            has_legal = any(keyword in content for keyword in legal_keywords)
            case_keywords = ['case', '案例', 'court', 'judgment', 'precedent', 'ruling']
            has_case = any(keyword in content for keyword in case_keywords)

            # 基础得分：向量相似度得分作为基础（权重0.3，降低向量相似度权重）
            score = vector_score * 0.3
            
            # 法律信号加权（权重0.2）
            if has_legal:
                score += 0.2
            # 案例信号加权（权重0.1）
            if has_case:
                score += 0.1
            
            # 检查文档元数据中的类别信息
            metadata = result.get('metadata', {}) or {}
            # 确保metadata是字典类型
            if not isinstance(metadata, dict):
                metadata = {}
            
            # 调试：检查metadata内容
            if not metadata and i < 3:  # 只打印前3个
                print(f"[DEBUG _evaluate_evidence] 警告：结果 {i} 的metadata为空，result keys: {list(result.keys())}")
                print(f"[DEBUG _evaluate_evidence] result内容: {result}")
            
            category = metadata.get('category', '')
            title = metadata.get('title', '')
            file_path = metadata.get('file', '')
            
            # 主题匹配加权（提高权重到0.3，如果匹配主题关键词）
            if topic_keywords:
                matched_keywords = [tk for tk in topic_keywords if tk in content]
                if matched_keywords:
                    # 匹配的关键词越多，得分越高
                    match_ratio = min(len(matched_keywords) / max(len(topic_keywords), 1), 1.0)
                    score += 0.3 * match_ratio
                
                # 检查category和title中的主题匹配（提高权重）
                if category:
                    category_lower = str(category).lower()
                    if any(kw in category_lower for kw in topic_keywords):
                        score += 0.15  # 提高category匹配权重
                    else:
                        # 负向过滤：如果category明确不匹配主题，降低得分
                        # 检查是否包含明显不相关的关键词
                        irrelevant_keywords = []
                        if any(kw in q for kw in ['employment', 'labor', 'employee', 'employer', '劳动', '雇佣', '员工', '雇主']):
                            irrelevant_keywords = ['saas', 'software', 'license', '软件', '许可', '订阅']
                        elif any(kw in q for kw in ['saas', 'software', 'license', '软件', '许可']):
                            irrelevant_keywords = ['employment', 'labor', 'employee', 'employer', '劳动', '雇佣', '员工', '雇主']
                        
                        if irrelevant_keywords and any(irr_kw in category_lower for irr_kw in irrelevant_keywords):
                            score -= 0.3  # 明显不相关，降低得分
                
                if title:
                    title_lower = str(title).lower()
                    if any(kw in title_lower for kw in topic_keywords):
                        score += 0.15  # 提高title匹配权重
                    else:
                        # 负向过滤：如果title明确不匹配主题，降低得分
                        irrelevant_keywords = []
                        if any(kw in q for kw in ['employment', 'labor', 'employee', 'employer', '劳动', '雇佣', '员工', '雇主']):
                            irrelevant_keywords = ['saas', 'software', 'license', '软件', '许可', '订阅']
                        elif any(kw in q for kw in ['saas', 'software', 'license', '软件', '许可']):
                            irrelevant_keywords = ['employment', 'labor', 'employee', 'employer', '劳动', '雇佣', '员工', '雇主']
                        
                        if irrelevant_keywords and any(irr_kw in title_lower for irr_kw in irrelevant_keywords):
                            score -= 0.3  # 明显不相关，降低得分

            score = min(score, 1.0)
            ev = {
                'content': content_full,
                'source': result.get('source', source_type) or source_type,
                'relevance_score': score,
                'provenance': source_type,
                'metadata': metadata,  # 确保传递完整的metadata
                'vector_score': vector_score  # 保留向量得分用于调试
            }
            evidence.append(ev)
        
        # 提高过滤阈值，确保只保留相关性较高的文档
        # 对于RAG检索结果，需要主题匹配度较高或向量得分非常高才保留
        evidence = [
            e for e in evidence 
            if e['relevance_score'] >= 0.4 or (e.get('vector_score', 0) >= 0.8)
        ]
        evidence.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # 返回前5个结果
        return evidence[:5]
    def _generate_uncertainty_note(self, confidence: str) -> str:
        notes = {
            'high': 'Based on reliable legal provisions, the answer has a high degree of confidence.',
            'medium': 'Based on external sources, the answers are for informational purposes only, further verification is recommended.',
            'low': 'Insufficient evidence supports the answer, it is recommended to consult a professional lawyer.'
        }
        return notes.get(confidence, '')
    def _get_timestamp(self) -> str:
        from datetime import datetime
        return datetime.now().isoformat()
    def get_recent_qa(self, n: int = 5) -> list:
        return self.qa_history[-n:] if self.qa_history else []


class ContractDraftingAgent:
    """
    合同起草智能体
    根据合同规格和模板生成合同。
    集成RAG检索以获取合同模板和相关法律条文。
    """
    def __init__(self, rag_retriever=None, reranker=None, llm_client=None, memory=None):
        from utils.llm_client import LLMClient
        self.rag_retriever = rag_retriever
        self.reranker = reranker
        self.llm_client = llm_client or LLMClient()
        self.memory = memory
        self.drafted_contracts = []

    def draft(self, contract_spec: dict, contract_type: str, jurisdiction: str = 'Singapore') -> dict:
        """
        起草合同
        Args:
            contract_spec: 合同规格
            contract_type: 合同类型
            jurisdiction: 司法管辖区
        Returns:
            生成的合同
        """
        # 从RAG检索相关模板和法律条文
        retrieved_templates = self._retrieve_template_from_rag(contract_type, jurisdiction)
        retrieved_laws = self._retrieve_related_laws(contract_type, jurisdiction)
        
        # 合并检索到的模板
        template = self._merge_templates(retrieved_templates, contract_type)
        
        # 将检索到的模板和法律条文添加到合同规格中
        enhanced_spec = contract_spec.copy()
        if retrieved_templates:
            enhanced_spec['_retrieved_templates'] = [r.get('content', '')[:500] for r in retrieved_templates]
        if retrieved_laws:
            enhanced_spec['_retrieved_laws'] = [r.get('content', '')[:300] for r in retrieved_laws]
        
        # 如果有检索到的模板，将其添加到规格中
        if template and len(retrieved_templates) > 0:
            enhanced_spec['_template_reference'] = template[:1000]  # 限制长度
        
        # 使用检索到的模板和法律条文生成合同
        contract_content = self.llm_client.generate_contract_draft(
            enhanced_spec, 
            contract_type
        )
        
        # 如果有检索到的模板，尝试使用模板作为基础
        if template and len(retrieved_templates) > 0 and not contract_content:
            contract_content = self._generate_contract(enhanced_spec, template, contract_type)
        todo_placeholders = self._extract_todo_placeholders(contract_content)
        contract_doc = {
            'content': contract_content,
            'type': contract_type,
            'jurisdiction': jurisdiction,
            'todo_placeholders': todo_placeholders,
            'metadata': {
                'parties': contract_spec.get('parties', []),
                'draft_date': self._get_current_date(),
                'retrieved_templates': len(retrieved_templates),
                'retrieved_laws': len(retrieved_laws)
            }
        }
        self.drafted_contracts.append(contract_doc)
        return contract_doc
    def _retrieve_template_from_rag(self, contract_type: str, jurisdiction: str) -> list:
        """从RAG检索合同模板"""
        if not self.rag_retriever:
            return []
        
        # 构建查询以检索模板
        query = f"{contract_type} contract template {jurisdiction}"
        
        # 检索模板（从template来源）
        template_results = self.rag_retriever.search_by_source(query, source='template', top_k=3)
        
        # 可选：使用reranker提高质量
        if self.reranker and template_results:
            template_results = self.reranker.rerank(query, template_results, top_k=3)
        
        return template_results
    
    def _retrieve_related_laws(self, contract_type: str, jurisdiction: str) -> list:
        """检索相关法律条文"""
        if not self.rag_retriever:
            return []
        
        # 构建查询以检索相关法律
        query = f"{contract_type} legal requirements {jurisdiction} contract law"
        
        # 检索法律条文（从legal来源）
        law_results = self.rag_retriever.search_by_source(query, source='legal', top_k=5)
        
        # 可选：使用reranker提高质量
        if self.reranker and law_results:
            law_results = self.reranker.rerank(query, law_results, top_k=5)
        
        return law_results
    
    def _merge_templates(self, retrieved_templates: list, contract_type: str) -> str:
        """合并检索到的模板，如果检索失败则使用默认模板"""
        if retrieved_templates:
            # 合并所有检索到的模板内容
            template_parts = []
            for result in retrieved_templates:
                content = result.get('content', '')
                if content:
                    template_parts.append(content)
            if template_parts:
                return '\n\n---\n\n'.join(template_parts)
        
        # 如果没有检索到模板，使用默认模板
        return self._get_default_template(contract_type)
    def _get_default_template(self, contract_type: str) -> str:
        templates = {
            'distribution_agency': self._get_distribution_agency_template(),
            'employment': self._get_employment_template(),
            'engineering': self._get_engineering_template(),
            'guarantee_security': self._get_guarantee_security_template(),
            'ip_licensing': self._get_ip_licensing_template(),
            'financing': self._get_financing_template(),
            'msa_service': self._get_service_template(),  # 复用服务协议模板
            'nda': self._get_nda_template(),
            'partnership_jv': self._get_partnership_template(),  # 复用合伙协议模板
            'saas_software': self._get_saas_software_template(),
            'sale_purchase': self._get_sale_purchase_template(),
            'shareholders': self._get_shareholders_template()
        }
        return templates.get(contract_type, self._get_general_template())
    def _generate_contract(self, spec: dict, template: str, contract_type: str) -> str:
        content = template
        replacements = {
            '{PARTIES}': self._format_parties(spec.get('parties', [])),
            '{DURATION}': spec.get('duration', '待确认'),
            '{AMOUNT}': spec.get('amount', spec.get('contract_price', spec.get('loan_amount', spec.get('subscription_fee', spec.get('price', '待确认'))))),
            '{DATE}': self._get_current_date(),
            '{SERVICES}': spec.get('services', '待确认'),
            '{POSITION}': spec.get('position', '待确认'),
            '{SALARY}': spec.get('salary', '待确认'),
            '{PURPOSE}': spec.get('purpose', '待确认'),
            '{SCOPE}': spec.get('scope', '商业机密信息'),
            # 新增字段
            '{TERRITORY}': spec.get('territory', '待确认'),
            '{PRODUCTS}': spec.get('products', '待确认'),
            '{COMMISSION}': spec.get('commission', '待确认'),
            '{PROJECT_SCOPE}': spec.get('project_scope', '待确认'),
            '{WARRANTY}': spec.get('warranty', '待确认'),
            '{PRINCIPAL_DEBT}': spec.get('principal_debt', '待确认'),
            '{GUARANTEE_AMOUNT}': spec.get('guarantee_amount', '待确认'),
            '{SECURITY_TYPE}': spec.get('security_type', '待确认'),
            '{IP_TYPE}': spec.get('ip_type', '待确认'),
            '{LICENSE_SCOPE}': spec.get('license_scope', '待确认'),
            '{ROYALTY}': spec.get('royalty', '待确认'),
            '{INTEREST_RATE}': spec.get('interest_rate', '待确认'),
            '{REPAYMENT_TERMS}': spec.get('repayment_terms', '待确认'),
            '{SECURITY}': spec.get('security', '待确认'),
            '{SOFTWARE_DESCRIPTION}': spec.get('software_description', '待确认'),
            '{LICENSE_TYPE}': spec.get('license_type', '待确认'),
            '{SUBSCRIPTION_FEE}': spec.get('subscription_fee', '待确认'),
            '{SUPPORT}': spec.get('support', '待确认'),
            '{GOODS_DESCRIPTION}': spec.get('goods_description', '待确认'),
            '{DELIVERY_TERMS}': spec.get('delivery_terms', '待确认'),
            '{SHAREHOLDING_RATIO}': spec.get('shareholding_ratio', '待确认'),
            '{BOARD_COMPOSITION}': spec.get('board_composition', '待确认'),
            '{DIVIDEND_POLICY}': spec.get('dividend_policy', '待确认'),
            '{TRANSFER_RESTRICTIONS}': spec.get('transfer_restrictions', '待确认')
        }
        for placeholder, value in replacements.items():
            content = content.replace(placeholder, str(value))
        return content
    def _extract_todo_placeholders(self, content: str) -> list:
        todos = []
        import re
        matches = re.finditer(r'\{([^}]+)\}', content)
        for match in matches:
            todos.append({
                'placeholder': match.group(0),
                'field': match.group(1),
                'location': match.start()
            })
        if '待确认' in content:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if '待确认' in line:
                    todos.append({
                        'line': i + 1,
                        'content': line.strip(),
                        'type': 'pending_confirmation'
                    })
        return todos
    def _format_parties(self, parties: list) -> str:
        if not parties:
            return '甲方：__________\n\n乙方：__________'
        elif len(parties) == 2:
            return f'甲方：{parties[0]}\n\n乙方：{parties[1]}'
        else:
            return '\n'.join([f'{party}' for party in parties])
    def _get_current_date(self) -> str:
        from datetime import datetime
        return datetime.now().strftime('%Y年%m月%d日')
    def _get_nda_template(self) -> str:
        return """# 保密协议 (Non-Disclosure Agreement)\n\n## 缔约双方\n{PARTIES}\n\n## 1. 定义\n本协议中的"保密信息"指：{SCOPE}\n\n## 2. 保密义务\n接收方同意对披露方提供的所有保密信息进行保密。\n\n## 3. 期限\n本协议有效期自签署之日起{PURPOSE}，或直至该保密信息不再构成保密信息为止。\n\n## 4. 适用法律\n本协议受新加坡法律管辖。\n\n## 5. 争议解决\n任何争议应提交新加坡国际仲裁中心仲裁。\n\n签署日期：{DATE}\n\n甲方：_____________________\n乙方：_____________________"""
    def _get_service_template(self) -> str:
        return """# 服务协议 (Service Agreement)\n\n## 缔约双方\n{PARTIES}\n\n## 1. 服务内容\n服务提供方同意提供以下服务：{SERVICES}\n\n## 2. 服务期限\n服务期限为{DURATION}。\n\n## 3. 报酬\n服务费用为{AMOUNT}。\n\n## 4. 付款方式\n双方同意按以下方式付款：（待确认）\n\n## 5. 终止条款\n任一方可在书面通知对方后终止本协议。\n\n## 6. 适用法律\n本协议受新加坡法律管辖。\n\n签署日期：{DATE}\n\n甲方：_____________________\n乙方：_____________________"""
    def _get_partnership_template(self) -> str:
        return """# 合伙协议 (Partnership Agreement)\n\n## 缔约双方\n{PARTIES}\n\n## 1. 合伙企业名称\n合伙企业名称：（待确认）\n\n## 2. 合伙期限\n合伙期限为{DURATION}。\n\n## 3. 出资比例\n各方出资比例如下：（待确认）\n\n## 4. 利润分配\n利润按出资比例分配。\n\n## 5. 管理机构\n合伙企业由各合伙人共同管理。\n\n## 6. 适用法律\n本协议受新加坡法律管辖。\n\n签署日期：{DATE}\n\n合伙人签字：\n_____________________\n_____________________"""
    def _get_employment_template(self) -> str:
        return """# 雇佣合同 (Employment Contract)\n\n## 缔约双方\n雇主：{PARTIES}\n雇员：（待确认）\n\n## 1. 职位\n雇员职位：{POSITION}\n\n## 2. 薪酬\n基本工资：{SALARY}\n\n## 3. 工作地点\n工作地点：（待确认）\n\n## 4. 工作时间\n工作时间：（待确认）\n\n## 5. 试用期\n试用期为（待确认）\n\n## 6. 终止条款\n任一方可提前一个月书面通知终止雇佣关系。\n\n## 7. 适用法律\n本协议受新加坡法律管辖。\n\n签署日期：{DATE}\n\n雇主：_____________________\n雇员：_____________________"""
    def _get_distribution_agency_template(self) -> str:
        return """# 分销代理协议 (Distribution Agency Agreement)\n\n## 缔约双方\n{PARTIES}\n\n## 1. 代理区域\n代理区域：（待确认）\n\n## 2. 代理产品\n代理产品：（待确认）\n\n## 3. 代理期限\n代理期限为{DURATION}。\n\n## 4. 佣金\n佣金比例：（待确认）\n\n## 5. 终止条款\n任一方可在书面通知对方后终止本协议。\n\n## 6. 适用法律\n本协议受新加坡法律管辖。\n\n签署日期：{DATE}\n\n甲方：_____________________\n乙方：_____________________"""
    def _get_engineering_template(self) -> str:
        return """# 工程合同 (Engineering Contract)\n\n## 缔约双方\n{PARTIES}\n\n## 1. 工程范围\n工程范围：（待确认）\n\n## 2. 合同价格\n合同总价为{AMOUNT}。\n\n## 3. 工程期限\n工程期限为{DURATION}。\n\n## 4. 付款方式\n付款方式：（待确认）\n\n## 5. 保修期\n保修期为（待确认）\n\n## 6. 适用法律\n本协议受新加坡法律管辖。\n\n签署日期：{DATE}\n\n甲方：_____________________\n乙方：_____________________"""
    def _get_guarantee_security_template(self) -> str:
        return """# 担保保证协议 (Guarantee Security Agreement)\n\n## 缔约双方\n{PARTIES}\n\n## 1. 主债务\n主债务金额：（待确认）\n\n## 2. 担保金额\n担保金额：（待确认）\n\n## 3. 担保期限\n担保期限为{DURATION}。\n\n## 4. 担保方式\n担保方式：（待确认）\n\n## 5. 适用法律\n本协议受新加坡法律管辖。\n\n签署日期：{DATE}\n\n担保人：_____________________\n债权人：_____________________"""
    def _get_ip_licensing_template(self) -> str:
        return """# 知识产权许可协议 (IP Licensing Agreement)\n\n## 缔约双方\n{PARTIES}\n\n## 1. 知识产权类型\n知识产权类型：（待确认）\n\n## 2. 许可范围\n许可范围：（待确认）\n\n## 3. 许可区域\n许可区域：（待确认）\n\n## 4. 许可期限\n许可期限为{DURATION}。\n\n## 5. 许可费用\n许可费用：（待确认）\n\n## 6. 适用法律\n本协议受新加坡法律管辖。\n\n签署日期：{DATE}\n\n许可方：_____________________\n被许可方：_____________________"""
    def _get_financing_template(self) -> str:
        return """# 融资协议 (Financing Agreement)\n\n## 缔约双方\n{PARTIES}\n\n## 1. 融资金额\n融资金额为{AMOUNT}。\n\n## 2. 利率\n利率：（待确认）\n\n## 3. 还款方式\n还款方式：（待确认）\n\n## 4. 融资期限\n融资期限为{DURATION}。\n\n## 5. 担保\n担保方式：（待确认）\n\n## 6. 适用法律\n本协议受新加坡法律管辖。\n\n签署日期：{DATE}\n\n贷款人：_____________________\n借款人：_____________________"""
    def _get_saas_software_template(self) -> str:
        return """# SaaS软件许可协议 (SaaS Software License Agreement)\n\n## 缔约双方\n{PARTIES}\n\n## 1. 软件描述\n软件描述：（待确认）\n\n## 2. 许可类型\n许可类型：（待确认）\n\n## 3. 订阅费用\n订阅费用为{AMOUNT}。\n\n## 4. 订阅期限\n订阅期限为{DURATION}。\n\n## 5. 技术支持\n技术支持：（待确认）\n\n## 6. 适用法律\n本协议受新加坡法律管辖。\n\n签署日期：{DATE}\n\n许可方：_____________________\n被许可方：_____________________"""
    def _get_sale_purchase_template(self) -> str:
        return """# 买卖协议 (Sale and Purchase Agreement)\n\n## 缔约双方\n{PARTIES}\n\n## 1. 货物描述\n货物描述：（待确认）\n\n## 2. 价格\n价格为{AMOUNT}。\n\n## 3. 交付条款\n交付条款：（待确认）\n\n## 4. 付款方式\n付款方式：（待确认）\n\n## 5. 保修\n保修期：（待确认）\n\n## 6. 适用法律\n本协议受新加坡法律管辖。\n\n签署日期：{DATE}\n\n卖方：_____________________\n买方：_____________________"""
    def _get_shareholders_template(self) -> str:
        return """# 股东协议 (Shareholders Agreement)\n\n## 缔约双方\n{PARTIES}\n\n## 1. 持股比例\n持股比例：（待确认）\n\n## 2. 董事会组成\n董事会组成：（待确认）\n\n## 3. 分红政策\n分红政策：（待确认）\n\n## 4. 股权转让限制\n股权转让限制：（待确认）\n\n## 5. 适用法律\n本协议受新加坡法律管辖。\n\n签署日期：{DATE}\n\n股东签字：\n_____________________\n_____________________"""
    def _get_general_template(self) -> str:
        return """# 合同 (Contract Agreement)\n\n## 缔约双方\n{PARTIES}\n\n## 1. 合同标的\n{AMOUNT}\n\n## 2. 合同期限\n{DURATION}\n\n## 3. 其他条款\n（待根据具体情况补充）\n\n## 4. 适用法律\n本协议受新加坡法律管辖。\n\n签署日期：{DATE}\n\n甲方：_____________________\n乙方：_____________________"""


class ContractReviewAgent:
    """
    合同审阅智能体
    实现合同内容风险自动审查与点评。
    支持合同智能分析、风险识别、缺失检测、建议生成等。
    集成RAG检索以获取相关法律条文和案例进行审查参考。
    """
    # 常见风险点
    RISK_PATTERNS = [
        '无限责任',
        '不明确的条款',
        '缺少争议解决机制',
        '缺少终止条件',
        '不公平的违约条款',
        '缺少法律适用声明'
    ]
    # 标准条款检查项
    STANDARD_CLAUSES = [
        'parties',
        'subject_matter',
        'consideration',
        'duration',
        'termination',
        'dispute_resolution',
        'governing_law'
    ]
    def __init__(self, rag_retriever=None, reranker=None, llm_client=None, memory=None):
        """初始化合同审阅智能体"""
        from utils.llm_client import LLMClient
        self.rag_retriever = rag_retriever
        self.reranker = reranker
        self.llm_client = llm_client or LLMClient()
        self.memory = memory
        self.reviewed_contracts = []

    def review(self, contract: dict, contract_spec: dict = None) -> dict:
        """
        审阅合同
        Args:
            contract: 合同内容
            contract_spec: 合同规格（用于对比）
        Returns:
            审阅报告
        """
        content = contract.get('content', '')
        contract_type = contract.get('type', 'general')
        
        # 从RAG检索相关法律条文和案例
        relevant_laws = self._retrieve_relevant_laws(content, contract_type)
        relevant_cases = self._retrieve_relevant_cases(content, contract_type)
        
        analysis = self._analyze_contract(content)
        risk_points = self._identify_risks(content, relevant_laws)
        missing_clauses = self._check_standard_clauses(content)
        recommendations = self._generate_recommendations(
            content, risk_points, missing_clauses, relevant_laws
        )
        
        # 将检索到的法律条文和案例添加到审查内容中
        enhanced_content = content
        if relevant_laws:
            law_text = "\n\n=== 相关法律条文参考 ===\n"
            for i, law in enumerate(relevant_laws[:3], 1):
                law_content = law.get('content', '')[:300]
                law_title = law.get('metadata', {}).get('title', '未知')
                law_text += f"\n[{i}] {law_title}:\n{law_content}...\n"
            enhanced_content = content + law_text
        
        if relevant_cases:
            case_text = "\n\n=== 相关案例参考 ===\n"
            for i, case in enumerate(relevant_cases[:2], 1):
                case_content = case.get('content', '')[:300]
                case_title = case.get('metadata', {}).get('title', '未知')
                case_text += f"\n[{i}] {case_title}:\n{case_content}...\n"
            enhanced_content = enhanced_content + case_text
        
        # 使用检索到的法律条文和案例增强LLM审查
        llm_review = self.llm_client.generate_contract_review(
            enhanced_content, 
            contract_type
        )
        
        report = {
            'analysis': analysis,
            'risk_points': risk_points,
            'missing_clauses': missing_clauses,
            'recommendations': recommendations,
            'severity': self._calculate_severity(risk_points),
            'confidence': self._calculate_confidence(analysis, len(relevant_laws)),
            'llm_review': llm_review,
            'legal_references': relevant_laws[:3],  # 只保留前3个相关法律
            'case_references': relevant_cases[:2]  # 只保留前2个相关案例
        }
        self.reviewed_contracts.append({
            'contract': contract,
            'report': report,
            'timestamp': self._get_timestamp()
        })
        return report
    
    def _retrieve_relevant_laws(self, contract_content: str, contract_type: str) -> list:
        """检索相关法律条文"""
        if not self.rag_retriever:
            return []
        
        # 从合同内容中提取关键概念进行检索
        keywords = self._extract_keywords_from_contract(contract_content)
        query = f"{contract_type} {' '.join(keywords[:5])} legal requirements"
        
        # 检索法律条文
        law_results = self.rag_retriever.search_by_source(query, source='legal', top_k=5)
        
        # 使用reranker提高相关性
        if self.reranker and law_results:
            law_results = self.reranker.rerank(query, law_results, top_k=5)
        
        return law_results
    
    def _retrieve_relevant_cases(self, contract_content: str, contract_type: str) -> list:
        """检索相关案例"""
        if not self.rag_retriever:
            return []
        
        # 从合同内容中提取关键概念进行检索
        keywords = self._extract_keywords_from_contract(contract_content)
        query = f"{contract_type} {' '.join(keywords[:5])} case law precedent"
        
        # 检索案例
        case_results = self.rag_retriever.search_by_source(query, source='case', top_k=3)
        
        # 使用reranker提高相关性
        if self.reranker and case_results:
            case_results = self.reranker.rerank(query, case_results, top_k=3)
        
        return case_results
    
    def _extract_keywords_from_contract(self, content: str) -> list:
        """从合同内容中提取关键词"""
        import re
        # 提取可能的法律关键词
        keywords = []
        legal_terms = ['termination', 'breach', 'liability', 'dispute', 'arbitration', 
                       'indemnity', 'warranty', 'representation', 'payment', 'delivery',
                       '终止', '违约', '责任', '争议', '仲裁', '赔偿', '保证', '付款', '交付']
        content_lower = content.lower()
        for term in legal_terms:
            if term in content_lower:
                keywords.append(term)
        return keywords[:10]  # 限制为前10个关键词
    
    # 其余所有私有方法（_analyze_contract, _identify_risks, ...）保持原样迁移
    def _analyze_contract(self, content: str) -> dict:
        # ... 请完整迁移原代码 ...
        word_count = len(content.split())
        lines = content.split('\n')
        has_sections = any(line.startswith('#') for line in lines)
        has_parties = '甲方' in content or '乙方' in content or 'party' in content.lower()
        has_signature = '签字' in content or '签署' in content or 'signature' in content.lower()
        has_law = '法律' in content or 'law' in content.lower()
        return {
            'word_count': word_count,
            'lines': len(lines),
            'has_sections': has_sections,
            'has_parties': has_parties,
            'has_signature': has_signature,
            'has_governing_law': has_law,
            'structure_score': self._calculate_structure_score(has_sections, has_parties, has_signature, has_law)
        }
    def _identify_risks(self, content: str, legal_references: list = None) -> list:
        # ... 原方法迁移 ...
        risks = []
        for pattern in self.RISK_PATTERNS:
            if pattern in content:
                risks.append({
                    'type': 'pattern_match',
                    'pattern': pattern,
                    'severity': 'medium',
                    'description': f'发现风险模式：{pattern}',
                    'recommendation': f'建议检查并明确{pattern}相关条款'
                })
        if '待确认' in content:
            import re
            todos = re.findall(r'（待确认）|待确认', content)
            if todos:
                risks.append({
                    'type': 'incomplete',
                    'severity': 'high',
                    'count': len(todos),
                    'description': f'发现{len(todos)}处待确认项',
                    'recommendation': '建议完成所有待确认项后再签署'
                })
        if not any(word in content for word in ['期限', '期限', 'duration', 'term']):
            risks.append({
                'type': 'missing_clause',
                'severity': 'medium',
                'description': '缺少期限条款',
                'recommendation': '建议明确合同的生效期限'
            })
        if not any(word in content for word in ['争议', '仲裁', 'dispute', 'arbitration']):
            risks.append({
                'type': 'missing_clause',
                'severity': 'high',
                'description': '缺少争议解决条款',
                'recommendation': '建议添加争议解决条款（推荐新加坡国际仲裁中心）'
            })
        return risks
    def _check_standard_clauses(self, content: str) -> list:
        # ... 原方法迁移 ...
        missing = []
        checks = {
            'parties': ('甲方' in content or '乙方' in content or 'party' in content.lower()),
            'consideration': ('费用' in content or '金额' in content or 'amount' in content.lower() or '付款' in content),
            'duration': ('期限' in content or 'duration' in content.lower()),
            'termination': ('终止' in content or 'termination' in content.lower() or '解除' in content),
            'dispute_resolution': ('争议' in content or '仲裁' in content or 'dispute' in content.lower()),
            'governing_law': ('法律' in content or 'law' in content.lower() or '适用法' in content)
        }
        for clause, present in checks.items():
            if not present:
                missing.append(clause)
        return missing
    def _generate_recommendations(self, content: str, risks: list, missing: list, legal_references: list = None) -> list:
        # ... 原方法迁移 ...
        recommendations = []
        for risk in risks:
            if risk['severity'] in ['high', 'critical']:
                recommendations.append({
                    'priority': 'high',
                    'category': risk['type'],
                    'description': risk.get('recommendation', '需要关注'),
                    'action': self._suggest_action(risk['type'])
                })
        for clause in missing:
            recommendations.append({
                'priority': 'medium',
                'category': 'missing_clause',
                'description': f'建议添加{clause}条款',
                'action': self._suggest_action('missing_clause')
            })
        recommendations.append({
            'priority': 'low',
            'category': 'general',
            'description': '建议由专业律师最终审核',
            'action': 'consult_legal_counsel'
        })
        
        # 如果有检索到的法律条文，添加基于法律条文的建议
        if legal_references:
            for law_ref in legal_references[:2]:  # 只使用前2个法律条文
                law_content = law_ref.get('content', '')
                if law_content:
                    recommendations.append({
                        'priority': 'medium',
                        'category': 'legal_reference',
                        'description': f'相关法律条文：{law_content[:100]}...',
                        'action': 'review_legal_requirements',
                        'source': law_ref.get('metadata', {}).get('title', '')
                    })
        
        return recommendations
    def _suggest_action(self, risk_type: str) -> str:
        # ... 原方法迁移 ...
        actions = {
            'pattern_match': '重新审视并明确相关条款',
            'incomplete': '补充所有待确认项',
            'missing_clause': '添加相应标准条款',
            'unfair_term': '协商修改为更公平的条款'
        }
        return actions.get(risk_type, '需要进一步审查')
    def _calculate_structure_score(self, has_sections: bool, has_parties: bool, has_signature: bool, has_law: bool) -> float:
        score = 0.0
        if has_sections:
            score += 0.3
        if has_parties:
            score += 0.3
        if has_signature:
            score += 0.2
        if has_law:
            score += 0.2
        return score
    def _calculate_severity(self, risks: list) -> str:
        if not risks:
            return 'low'
        has_critical = any(r.get('severity') == 'critical' for r in risks)
        has_high = any(r.get('severity') == 'high' for r in risks)
        if has_critical:
            return 'critical'
        elif has_high:
            return 'high'
        else:
            return 'medium'
    def _calculate_confidence(self, analysis: dict, legal_ref_count: int = 0) -> float:
        base_confidence = 0.7
        structure_score = analysis.get('structure_score', 0)
        # 如果有法律条文参考，提高置信度
        legal_boost = min(legal_ref_count * 0.05, 0.15)
        return base_confidence + structure_score * 0.3 + legal_boost
    def _get_timestamp(self) -> str:
        from datetime import datetime
        return datetime.now().isoformat()


class IntakeWizard:
    """
    信息采集与完整性校验器
    引导式问答采集合同要素并做完整性与一致性检查。
    """
    REQUIRED_FIELDS = {
        'distribution_agency': [
            'parties', 'territory', 'products', 'commission', 'duration', 'termination'
        ],
        'employment': [
            'parties', 'position', 'salary', 'benefits', 'working_hours', 'termination'
        ],
        'engineering': [
            'parties', 'project_scope', 'contract_price', 'duration', 'payment_terms', 'warranty'
        ],
        'guarantee_security': [
            'parties', 'principal_debt', 'guarantee_amount', 'guarantee_period', 'security_type'
        ],
        'ip_licensing': [
            'parties', 'ip_type', 'license_scope', 'territory', 'royalty', 'duration'
        ],
        'financing': [
            'parties', 'loan_amount', 'interest_rate', 'repayment_terms', 'duration', 'security'
        ],
        'msa_service': [
            'parties', 'services', 'payment_terms', 'duration', 'termination', 'sla'
        ],
        'nda': [
            'parties', 'purpose', 'duration', 'scope', 'confidential_information'
        ],
        'partnership_jv': [
            'parties', 'capital_contribution', 'profit_sharing', 'management', 'exit_clause'
        ],
        'saas_software': [
            'parties', 'software_description', 'license_type', 'subscription_fee', 'duration', 'support'
        ],
        'sale_purchase': [
            'parties', 'goods_description', 'price', 'delivery_terms', 'payment_terms', 'warranty'
        ],
        'shareholders': [
            'parties', 'shareholding_ratio', 'board_composition', 'dividend_policy', 'transfer_restrictions'
        ],
        'default': [
            'parties', 'subject_matter', 'consideration', 'effective_date'
        ]
    }
    
    # 核心必需字段：只要这些字段有了，就可以生成合同，其他字段可以使用默认值
    CORE_REQUIRED_FIELDS = {
        'distribution_agency': ['parties'],
        'employment': ['parties'],
        'engineering': ['parties'],
        'guarantee_security': ['parties'],
        'ip_licensing': ['parties'],
        'financing': ['parties'],
        'msa_service': ['parties', 'services'],  # 服务合同需要双方和服务描述
        'nda': ['parties'],
        'partnership_jv': ['parties'],
        'saas_software': ['parties'],
        'sale_purchase': ['parties'],
        'shareholders': ['parties'],
        'default': ['parties']
    }
    def __init__(self):
        self.current_session = {}
        self.sessions = {}  # 基于session_id的会话状态字典
    
    def collect(self, user_input: str, contract_type: str, session_context: dict = None) -> dict:
        """
        采集用户输入的信息，并返回缺失字段和一致性结果
        Args:
            user_input: 用户输入
            contract_type: 合同类型
            session_context: 可选，会话上下文，应包含 'session_id' 和可选的 'contract_spec'
        Returns:
            字段信息与缺失/一致性报告
        """
        session_id = session_context.get('session_id') if session_context else None
        
        # 基于session_id管理状态
        if session_id:
            if session_id in self.sessions:
                # 恢复该会话的状态
                self.current_session = self.sessions[session_id].copy()
            else:
                # 新会话，初始化状态
                self.current_session = {}
                # 如果session_context中有contract_spec，使用它
                if session_context and 'contract_spec' in session_context:
                    self.current_session = session_context['contract_spec'].copy()
        else:
            # 没有session_id，使用全局状态（向后兼容）
            if session_context and 'contract_spec' in session_context:
                self.current_session = session_context['contract_spec'].copy()
            else:
                self.current_session = self.current_session or {}
        
        # 提取信息并更新状态
        extracted_info = self._extract_info(user_input, contract_type)
        # print(f"[DEBUG collect] Extracted info: {extracted_info}")
        self.current_session.update(extracted_info)
        # print(f"[DEBUG collect] Current session after update: {self.current_session}")
        
        # 保存状态到sessions字典
        if session_id:
            self.sessions[session_id] = self.current_session.copy()
        
        required_fields = self._get_required_fields(contract_type)
        # print(f"[DEBUG collect] Required fields for {contract_type}: {required_fields}")
        missing_fields = self._check_completeness(self.current_session, required_fields)
        # print(f"[DEBUG collect] Missing fields: {missing_fields}")
        
        # 检查核心必需字段是否已收集
        core_required_fields = self._get_core_required_fields(contract_type)
        missing_core_fields = self._check_completeness(self.current_session, core_required_fields)
        core_complete = len(missing_core_fields) == 0
        # print(f"[DEBUG collect] Core required fields for {contract_type}: {core_required_fields}")
        # print(f"[DEBUG collect] Missing core fields: {missing_core_fields}")
        # print(f"[DEBUG collect] Core complete: {core_complete}")
        
        consistency_issues = self._check_consistency(self.current_session)
        return {
            'contract_spec': self.current_session.copy(),
            'missing_fields': missing_fields,
            'missing_core_fields': missing_core_fields,
            'core_complete': core_complete,
            'consistency_issues': consistency_issues,
            'is_complete': len(missing_fields) == 0,
            'next_question': self._generate_next_question(missing_fields, contract_type)
        }
    
    def get_session_state(self, session_id: str) -> dict:
        """获取指定会话的状态"""
        return self.sessions.get(session_id, {}).copy()
    
    def clear_session(self, session_id: str):
        """清除指定会话的状态"""
        if session_id in self.sessions:
            del self.sessions[session_id]
    def _extract_info(self, user_input: str, contract_type: str) -> dict:
        info = {}
        text_lower = user_input.lower()
        if any(word in text_lower for word in ['甲方', '乙方', '双方', 'party', 'parties']):
            extracted_parties = self._extract_parties(user_input)
            if extracted_parties:  # 只有成功提取到parties时才添加
                info['parties'] = extracted_parties
                # print(f"[DEBUG _extract_info] Extracted parties: {extracted_parties}")
            else:
                print(f"[DEBUG _extract_info] Failed to extract parties from: {user_input}")
        if any(word in text_lower for word in ['金额', '价格', 'amount', 'price', 'fee']):
            info['amount'] = self._extract_amount(text_lower)
        if any(word in text_lower for word in ['期限', '时间', 'duration', 'period', 'term']):
            info['duration'] = self._extract_duration(text_lower)
        if contract_type == 'nda':
            info.update(self._extract_nda_info(text_lower))
        elif contract_type == 'msa_service':
            info.update(self._extract_service_info(text_lower))
        elif contract_type == 'partnership_jv':
            info.update(self._extract_partnership_info(text_lower))
        elif contract_type == 'employment':
            info.update(self._extract_employment_info(text_lower))
        elif contract_type == 'distribution_agency':
            info.update(self._extract_distribution_agency_info(text_lower))
        elif contract_type == 'engineering':
            info.update(self._extract_engineering_info(text_lower))
        elif contract_type == 'guarantee_security':
            info.update(self._extract_guarantee_security_info(text_lower))
        elif contract_type == 'ip_licensing':
            info.update(self._extract_ip_licensing_info(text_lower))
        elif contract_type == 'financing':
            info.update(self._extract_financing_info(text_lower))
        elif contract_type == 'saas_software':
            info.update(self._extract_saas_software_info(text_lower))
        elif contract_type == 'sale_purchase':
            info.update(self._extract_sale_purchase_info(text_lower))
        elif contract_type == 'shareholders':
            info.update(self._extract_shareholders_info(text_lower))
        return info
    def _extract_parties(self, text: str) -> list:
        """解析甲乙双方名称，支持格式如：甲方：A，乙方：B / 甲方:A 乙方:B / Party A: A; Party B: B"""
        import re
        parties: list = []

        # 尝试中文格式
        m_a = re.search(r'甲方\s*[:：]\s*([^，,\n;；]+)', text)
        m_b = re.search(r'乙方\s*[:：]\s*([^，,\n;；]+)', text)
        if m_a and m_b:
            a = m_a.group(1).strip()
            b = m_b.group(1).strip()
            if a:
                parties.append(a)
            if b:
                parties.append(b)
            if parties:
                return parties[:2]

        # 尝试英文 Party A/B 格式
        # 支持多种格式：Party A: XXX, Party B: XXX / Party A: XXX; Party B: XXX / Party A: XXX. Party B: XXX.
        # 改进正则表达式：更灵活地匹配Party A和Party B的值
        # 匹配模式：Party A: 后面跟着非空白字符，直到遇到逗号、分号、句号、换行符或Party B
        m_pa = re.search(r'party\s*a\s*[:：\-]?\s*([^，,\n;；.]+?)(?:\s*[,;.]|party\s*b|$)', text, re.IGNORECASE)
        m_pb = re.search(r'party\s*b\s*[:：\-]?\s*([^，,\n;；.]+?)(?:\s*[,;.]|$)', text, re.IGNORECASE)
        if m_pa and m_pb:
            a = m_pa.group(1).strip().rstrip(',.')  # 去除末尾的逗号和句号
            b = m_pb.group(1).strip().rstrip(',.')
            if a and b:  # 确保提取到的值不为空
                parties = [a, b]
                # print(f"[DEBUG _extract_parties] Extracted parties from English format: {parties}")
                return parties[:2]
        # 如果上面的正则表达式没有匹配到，尝试更简单的模式
        # 匹配：Party A: 任意字符直到逗号或Party B
        m_pa_simple = re.search(r'party\s*a\s*[:：]\s*([^,]+?)(?:,|party\s*b)', text, re.IGNORECASE)
        m_pb_simple = re.search(r'party\s*b\s*[:：]\s*([^,]+?)(?:[,.]|$)', text, re.IGNORECASE)
        if m_pa_simple and m_pb_simple:
            a = m_pa_simple.group(1).strip().rstrip(',.')
            b = m_pb_simple.group(1).strip().rstrip(',.')
            if a and b:
                parties = [a, b]
                # print(f"[DEBUG _extract_parties] Extracted parties from simple format: {parties}")
                return parties[:2]

        # 回退：若文本同时包含“甲方”和“乙方”，但未能解析名称，则返回占位名称
        if ('甲方' in text) and ('乙方' in text):
            return ['甲方', '乙方']
        if re.search(r'party\s*a', text, re.IGNORECASE) and re.search(r'party\s*b', text, re.IGNORECASE):
            return ['Party A', 'Party B']

        return parties
    def _extract_amount(self, text: str):
        import re
        amounts = re.findall(r'\d+(?:\.\d+)?(?:\s*[万|千|hundred|thousand|million])?', text)
        return amounts[0] if amounts else None
    def _extract_duration(self, text: str):
        import re
        duration_keywords = ['年', '月', '天', 'year', 'month', 'day', 'week']
        for keyword in duration_keywords:
            if keyword in text:
                matches = re.findall(r'(\d+)\s*' + keyword, text)
                if matches:
                    return f"{matches[0]} {keyword}"
        return None
    def _extract_nda_info(self, text: str) -> dict:
        info = {}
        if '保密信息' in text or 'confidential' in text:
            info['scope'] = '商业信息、技术资料'
        if '目的' in text or 'purpose' in text:
            info['purpose'] = self._extract_text_after_keyword(text, '目的', 'purpose')
        return info
    def _extract_service_info(self, text: str) -> dict:
        info = {}
        if '服务' in text or 'service' in text:
            info['services'] = self._extract_text_after_keyword(text, '服务', 'service')
        return info
    def _extract_partnership_info(self, text: str) -> dict:
        info = {}
        if '合伙' in text or 'partnership' in text:
            info['type'] = '一般合伙'
        return info
    def _extract_employment_info(self, text: str) -> dict:
        info = {}
        if '职位' in text or 'position' in text:
            info['position'] = self._extract_text_after_keyword(text, '职位', 'position')
        return info
    def _extract_distribution_agency_info(self, text: str) -> dict:
        info = {}
        if '区域' in text or 'territory' in text:
            info['territory'] = self._extract_text_after_keyword(text, '区域', 'territory')
        if '产品' in text or 'product' in text:
            info['products'] = self._extract_text_after_keyword(text, '产品', 'product')
        if '佣金' in text or 'commission' in text:
            info['commission'] = self._extract_text_after_keyword(text, '佣金', 'commission')
        return info
    def _extract_engineering_info(self, text: str) -> dict:
        info = {}
        if '工程' in text or 'project' in text or '工程范围' in text:
            info['project_scope'] = self._extract_text_after_keyword(text, '工程范围', 'project scope')
        if '价格' in text or 'price' in text or '合同价格' in text:
            info['contract_price'] = self._extract_text_after_keyword(text, '合同价格', 'contract price')
        if '保修' in text or 'warranty' in text:
            info['warranty'] = self._extract_text_after_keyword(text, '保修', 'warranty')
        return info
    def _extract_guarantee_security_info(self, text: str) -> dict:
        info = {}
        if '主债务' in text or 'principal debt' in text:
            info['principal_debt'] = self._extract_text_after_keyword(text, '主债务', 'principal debt')
        if '担保金额' in text or 'guarantee amount' in text:
            info['guarantee_amount'] = self._extract_text_after_keyword(text, '担保金额', 'guarantee amount')
        if '担保方式' in text or 'security type' in text:
            info['security_type'] = self._extract_text_after_keyword(text, '担保方式', 'security type')
        return info
    def _extract_ip_licensing_info(self, text: str) -> dict:
        info = {}
        if '知识产权' in text or 'ip' in text or 'intellectual property' in text:
            info['ip_type'] = self._extract_text_after_keyword(text, '知识产权类型', 'ip type')
        if '许可范围' in text or 'license scope' in text:
            info['license_scope'] = self._extract_text_after_keyword(text, '许可范围', 'license scope')
        if '许可区域' in text or 'territory' in text:
            info['territory'] = self._extract_text_after_keyword(text, '许可区域', 'territory')
        if '许可费用' in text or 'royalty' in text:
            info['royalty'] = self._extract_text_after_keyword(text, '许可费用', 'royalty')
        return info
    def _extract_financing_info(self, text: str) -> dict:
        info = {}
        if '融资金额' in text or 'loan amount' in text:
            info['loan_amount'] = self._extract_text_after_keyword(text, '融资金额', 'loan amount')
        if '利率' in text or 'interest rate' in text:
            info['interest_rate'] = self._extract_text_after_keyword(text, '利率', 'interest rate')
        if '还款' in text or 'repayment' in text:
            info['repayment_terms'] = self._extract_text_after_keyword(text, '还款方式', 'repayment terms')
        if '担保' in text or 'security' in text:
            info['security'] = self._extract_text_after_keyword(text, '担保', 'security')
        return info
    def _extract_saas_software_info(self, text: str) -> dict:
        info = {}
        if '软件' in text or 'software' in text:
            info['software_description'] = self._extract_text_after_keyword(text, '软件描述', 'software description')
        if '许可类型' in text or 'license type' in text:
            info['license_type'] = self._extract_text_after_keyword(text, '许可类型', 'license type')
        if '订阅费用' in text or 'subscription fee' in text:
            info['subscription_fee'] = self._extract_text_after_keyword(text, '订阅费用', 'subscription fee')
        if '支持' in text or 'support' in text:
            info['support'] = self._extract_text_after_keyword(text, '技术支持', 'support')
        return info
    def _extract_sale_purchase_info(self, text: str) -> dict:
        info = {}
        if '货物' in text or 'goods' in text:
            info['goods_description'] = self._extract_text_after_keyword(text, '货物描述', 'goods description')
        if '交付' in text or 'delivery' in text:
            info['delivery_terms'] = self._extract_text_after_keyword(text, '交付条款', 'delivery terms')
        if '保修' in text or 'warranty' in text:
            info['warranty'] = self._extract_text_after_keyword(text, '保修', 'warranty')
        return info
    def _extract_shareholders_info(self, text: str) -> dict:
        info = {}
        if '持股' in text or 'shareholding' in text:
            info['shareholding_ratio'] = self._extract_text_after_keyword(text, '持股比例', 'shareholding ratio')
        if '董事会' in text or 'board' in text:
            info['board_composition'] = self._extract_text_after_keyword(text, '董事会组成', 'board composition')
        if '分红' in text or 'dividend' in text:
            info['dividend_policy'] = self._extract_text_after_keyword(text, '分红政策', 'dividend policy')
        if '转让' in text or 'transfer' in text:
            info['transfer_restrictions'] = self._extract_text_after_keyword(text, '股权转让限制', 'transfer restrictions')
        return info
    def _extract_text_after_keyword(self, text: str, *keywords) -> str:
        for keyword in keywords:
            if keyword in text:
                words = text.split(keyword)
                if len(words) > 1:
                    return words[1].strip()
        return ''
    def _get_required_fields(self, contract_type: str) -> list:
        return self.REQUIRED_FIELDS.get(contract_type, self.REQUIRED_FIELDS['default'])
    
    def _get_core_required_fields(self, contract_type: str) -> list:
        return self.CORE_REQUIRED_FIELDS.get(contract_type, self.CORE_REQUIRED_FIELDS['default'])
    def _check_completeness(self, contract_spec: dict, required_fields: list) -> list:
        missing = []
        for field in required_fields:
            if field not in contract_spec or not contract_spec[field]:
                missing.append(field)
        return missing
    def _check_consistency(self, contract_spec: dict) -> list:
        issues = []
        # （可扩展一致性规则）
        if 'duration' in contract_spec and 'termination' in contract_spec:
            pass
        return issues
    def _generate_next_question(self, missing_fields: list, contract_type: str):
        if not missing_fields:
            return None
        field = missing_fields[0]
        
        # Force English only
        questions = {
            'parties': 'Please provide the names of both parties to the contract',
            'duration': 'Please provide the contract duration',
            'amount': 'Please provide the contract amount',
            'services': 'Please describe the services to be provided',
            'position': 'Please provide the job position',
            'salary': 'Please provide the salary information',
            'confidential_information': 'Please specify the scope of confidential information'
        }
        
        return questions.get(field, f'Please provide {field} information')
    def reset(self):
        self.current_session = {}


class IntentRouter:
    """
    意图路由器
    自动识别用户文本意图、合同类型与司法区域。
    """
    INTENT_KEYWORDS = {
        'contract_draft': [
            '起草合同', '生成合同', '创建合同', '制作合同', '写合同',
            '起草协议', '生成协议', '创建协议', '制作协议',
            'draft a contract', 'draft contract', 'create contract', 
            'generate contract', 'make contract', 'write contract',
            'draft an agreement', 'create agreement', 'generate agreement'
        ],
        'contract_review': [
            '审查', '审阅', '检查', 'review', 'examine',
            'check', '审核', 'revise', 'analyze'
        ],
        'legal_qa': [
            '咨询', '问题', '询问', 'legal',
            '法律', 'law', 'question', 'ask', 'what', 'how', 'why',
            'requirements', 'needed', 'needs', 'need', 'require'
        ]
    }
    CONTRACT_TYPE_KEYWORDS = {
        'distribution_agency': ['分销', '代理', 'distribution', 'agency', '经销商', '代理商'],
        'employment': ['雇佣', 'employment', '员工', '劳动合同', '劳动'],
        'engineering': ['工程', 'engineering', '建设', 'construction', '施工'],
        'guarantee_security': ['担保', '保证', 'guarantee', 'security', '抵押', '质押'],
        'ip_licensing': ['知识产权', '专利', '商标', '版权', 'intellectual property', 'ip', 'licensing', 'assignment', '授权'],
        'financing': ['融资', 'financing', '债务', 'debt', 'equity', '股权', '贷款', 'loan', '投资', 'investment'],
        'msa_service': ['服务', 'service', '劳务', 'msa', 'master service', '主服务'],
        'nda': ['保密', 'non-disclosure', 'nda', '保密协议'],
        'partnership_jv': ['合伙', 'partnership', '合作', 'joint venture', '合资', '合营'],
        'saas_software': ['saas', 'software', '软件', '软件许可', 'license', '云服务', 'cloud'],
        'sale_purchase': ['买卖', 'sale', 'purchase', '销售', '采购', '购买'],
        'shareholders': ['股东', 'shareholder', 'shareholders agreement', '股东协议']
    }
    JURISDICTION_KEYWORDS = {
        'Singapore': ['新加坡', 'singapore', 'sg']
    }
    def __init__(self):
        self.intent_history = []
    

    def route(self, user_input: str, session_id: str = None) -> dict:
        """
        路由用户输入到相应意图（审查/起草/法律咨询等）
        Args:
            user_input: 用户输入
            session_id: 会话id（可选）
        Returns:
            意图结构化分析结果
        """
        cleaned_input = user_input.lower().strip()
        intent = self._detect_intent(cleaned_input)
        contract_type = self._detect_contract_type(cleaned_input)
        jurisdiction = self._detect_jurisdiction(cleaned_input)
        task_context = self._extract_context(user_input)
        
        result = {
            'intent': intent,
            'contract_type': contract_type,
            'jurisdiction': jurisdiction,
            'task_context': task_context,
            'user_input': user_input,
            
            'confidence': self._calculate_confidence(intent, contract_type, jurisdiction)
        }
        self.intent_history.append(result)
        return result
    def _detect_intent(self, text: str) -> str:
        """检测意图，改进逻辑以避免误判"""
        # 先检查审查意图（优先级最高）
        if any(keyword in text for keyword in self.INTENT_KEYWORDS['contract_review']):
            return 'contract_review'
        
        # 如果是问句（包含what, how, why等疑问词），通常是法律咨询
        # 检查这个要放在起草意图之前，因为问句可能是法律咨询
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which', '什么', '如何', '为什么', '需要', 'needed', 'need', 'require', 'requirements']
        has_question_word = any(word in text for word in question_words)
        
        # 如果包含疑问词和咨询类词汇（what, how, requirements等），优先认为是法律咨询
        consulting_keywords = ['requirements', 'needed', 'needs', 'need', 'require', 'what', 'how', 'why', '咨询', '询问', '问题']
        has_consulting_keyword = any(kw in text for kw in consulting_keywords)
        
        if has_question_word and has_consulting_keyword:
            return 'legal_qa'
        
        # 检查起草意图 - 需要明确的起草动作词
        has_draft_action = any(action in text for action in [
            '起草', '生成', '创建', '制作', '写',
            'draft', 'create', 'generate', 'make', 'write', 'formulate', 'prepare'
        ])
        
        # 只有当有明确的起草动作词且包含合同相关词时，才认为是起草意图
        has_contract_word = any(word in text for word in ['合同', '协议', 'contract', 'agreement'])
        
        if has_draft_action and has_contract_word:
            return 'contract_draft'
        
        # 如果只是问句，也认为是法律咨询
        if has_question_word:
            return 'legal_qa'
        
        # 其他情况默认法律咨询
        return 'legal_qa'
    def _detect_contract_type(self, text: str):
        for contract_type, keywords in self.CONTRACT_TYPE_KEYWORDS.items():
            if any(keyword in text for keyword in keywords):
                return contract_type
        return None
    def _detect_jurisdiction(self, text: str) -> str:
        for jurisdiction, keywords in self.JURISDICTION_KEYWORDS.items():
            if any(keyword in text for keyword in keywords):
                return jurisdiction
        return 'Singapore'
    def _extract_context(self, text: str) -> dict:
        import re
        context = {
            'urgency': 'normal',
            'complexity': 'medium',
            'keywords': []
        }
        if any(word in text for word in ['紧急', 'urgent', 'asap', '尽快']):
            context['urgency'] = 'high'
        context['keywords'] = self._extract_keywords(text)
        return context
    def _extract_keywords(self, text: str) -> list:
        import re
        words = re.findall(r'\b\w+\b', text.lower())
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'from'}
        keywords = [w for w in words if len(w) > 3 and w not in stopwords]
        return keywords[:10]
    def _calculate_confidence(self, intent: str, contract_type: str, jurisdiction: str) -> float:
        confidence = 0.7
        if contract_type:
            confidence += 0.1
        if jurisdiction != 'Singapore':
            confidence += 0.1
        return min(confidence, 1.0)
    def get_recent_intents(self, n: int = 5) -> list:
        return self.intent_history[-n:] if self.intent_history else []

