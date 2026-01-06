
import gradio as gr
import os
from typing import Dict, List, Tuple

from agents.langchain_agents import (
    LegalQAAgent,
    ContractDraftingAgent,
    ContractReviewAgent,
    IntentRouter,
    IntakeWizard
)
from rag import VectorStore, LegalBERTEmbedder, RAGRetriever, Reranker, WebRetriever, BM25Retriever
import pickle
from utils import ConversationMemory, DataLoader, LLMClient

# 全局组件
intent_router = None
intake_wizard = None
contract_drafting_agent = None
contract_review_agent = None
legal_qa_agent = None
memory = None
rag_retriever = None
web_retriever = None
reranker = None


def _find_matching_embedder(target_dimension: int, preferred_model: str = None):
    """
    查找维度匹配的嵌入模型
    
    Args:
        target_dimension: 目标维度
        preferred_model: 优先尝试的模型名称
        
    Returns:
        匹配的 LegalBERTEmbedder 实例，如果找不到则返回 None
    """
    from rag.embedder import LegalBERTEmbedder
    
    # 默认优先使用 nlpaueb/legal-bert-base-uncased
    DEFAULT_PREFERRED = 'nlpaueb/legal-bert-base-uncased'
    
    # 常见模型的维度映射
    model_dimensions = {
        384: ['all-MiniLM-L6-v2', 'paraphrase-multilingual-MiniLM-L12-v2'],
        768: ['nlpaueb/legal-bert-base-uncased', 'sentence-transformers/all-mpnet-base-v2'],
        512: ['paraphrase-multilingual-MiniLM-L12-v2'],  # 某些版本可能是512
    }
    
    # 构建模型尝试列表，优先使用 nlpaueb/legal-bert-base-uncased
    models_to_try = []
    
    # 1. 如果指定了优先模型，先尝试它
    if preferred_model:
        models_to_try.append(preferred_model)
    
    # 2. 确保默认模型在最前面（如果还没有添加）
    if DEFAULT_PREFERRED not in models_to_try:
        models_to_try.append(DEFAULT_PREFERRED)
    
    # 3. 添加该维度常见的模型
    if target_dimension in model_dimensions:
        for model in model_dimensions[target_dimension]:
            if model not in models_to_try:
                models_to_try.append(model)
    
    # 4. 添加所有法律备选模型
    for model in LegalBERTEmbedder.LEGAL_MODEL_ALTERNATIVES:
        if model not in models_to_try:
            models_to_try.append(model)
    
    # 5. 最后添加通用备选模型
    models_to_try.extend(LegalBERTEmbedder.FALLBACK_MODELS)
    
    # 去除重复
    models_to_try = list(dict.fromkeys(models_to_try))
    
    # 尝试每个模型
    for model_name in models_to_try:
        try:
            embedder = LegalBERTEmbedder(model_name=model_name)
            # 确保模型已加载（通过获取维度）
            embedder_dim = embedder.get_dimension()
            
            if embedder_dim == target_dimension:
                print(f"✓ 找到维度匹配的模型: {model_name} (维度: {embedder_dim})")
                return embedder
            else:
                print(f"✗ 模型 {model_name} 维度不匹配: {embedder_dim} != {target_dimension}")
        except Exception as e:
            print(f"✗ 无法加载模型 {model_name}: {str(e)[:50]}")
            continue
    
    return None


def initialize_system():
    """初始化系统"""
    global intent_router, intake_wizard, contract_drafting_agent
    global contract_review_agent, legal_qa_agent, memory
    global rag_retriever, web_retriever, reranker
    
    print("正在初始化系统组件...")
    
    # 初始化LLM客户端
    try:
        llm_client = LLMClient()
        print("DeepSeek LLM客户端初始化成功")
    except Exception as e:
        print(f"LLM客户端初始化失败：{e}")
        llm_client = None
    
    # 优先检测是否已有索引（统一使用data/vectors路径）：
    vectors_dir = os.path.join('data', 'vectors')
    os.makedirs(vectors_dir, exist_ok=True)  # 确保目录存在
    vectors_index_path = os.path.join(vectors_dir, 'faiss.index')
    if os.path.exists(vectors_index_path):
        # 若已有索引，先加载 VectorStore 获取实际维度
        print("检测到已有向量索引，正在加载...")
        vector_store = VectorStore(index_path=vectors_index_path)
        index_dimension = vector_store.dimension
        print(f"索引维度: {index_dimension}")
        
        # 从配置中获取保存的模型名称
        model_name_from_index = None
        if isinstance(vector_store.config, dict):
            model_name_from_index = vector_store.config.get('embedder_model_name')
        
        # 尝试加载匹配维度的嵌入器
        embedder = None
        if model_name_from_index:
            print(f"尝试加载索引记录的模型: {model_name_from_index}")
            try:
                embedder = LegalBERTEmbedder(model_name=model_name_from_index)
                embedder_dim = embedder.get_dimension()
                
                if embedder_dim != index_dimension:
                    print(f"⚠ 警告: 模型维度 ({embedder_dim}) 与索引维度 ({index_dimension}) 不匹配")
                    embedder = None
                else:
                    print(f"✓ 成功加载模型，维度匹配: {embedder_dim}")
            except Exception as e:
                print(f"⚠ 无法加载记录的模型 {model_name_from_index}: {e}")
                embedder = None
        
        # 如果加载失败，优先尝试默认模型 nlpaueb/legal-bert-base-uncased
        if embedder is None:
            print(f"正在查找维度为 {index_dimension} 的嵌入模型...")
            # 优先尝试默认模型
            preferred_models = ['nlpaueb/legal-bert-base-uncased']
            if model_name_from_index and model_name_from_index not in preferred_models:
                preferred_models.insert(0, model_name_from_index)
            embedder = _find_matching_embedder(index_dimension, preferred_models[0] if preferred_models else None)
            
            if embedder is None:
                raise ValueError(
                    f"无法找到与索引维度 ({index_dimension}) 匹配的嵌入模型。\n"
                    f"建议:\n"
                    f"1. 删除现有索引并重新构建: 删除 {vectors_index_path} 和相关文件\n"
                    f"2. 或使用与索引构建时相同的嵌入模型\n"
                    f"3. 索引记录的模型: {model_name_from_index or '未知'}"
                )
    else:
        # 无索引时，使用默认模型 nlpaueb/legal-bert-base-uncased 创建嵌入器和向量库
        embedder = LegalBERTEmbedder(model_name='nlpaueb/legal-bert-base-uncased')
        vector_store = VectorStore(
            dimension=embedder.get_dimension(),
            index_path=vectors_index_path
        )
    
    # 注意：数据应通过 ingest_pdfs.py 导入向量库
    # 这里不再通过 DataLoader 构建向量库，因为数据已直接存储在向量库中
    print("提示：向量库数据应通过运行 'python utils/ingest_pdfs.py' 导入")
    print("      如果索引已存在，将直接使用现有索引")

    # 初始化RAG组件（仅在索引包含向量时启用）
    try:
        if hasattr(vector_store, 'index') and getattr(vector_store.index, 'ntotal', 0) > 0:
            # 检查metadata是否正确加载
            if not vector_store.metadata or len(vector_store.metadata) == 0:
                print("⚠ 警告：向量索引存在但metadata为空")
                print("   建议重新运行 'python utils/ingest_pdfs.py' 来重建索引")
            elif len(vector_store.metadata) != vector_store.index.ntotal:
                print(f"⚠ 警告：向量数量 ({vector_store.index.ntotal}) 与metadata数量 ({len(vector_store.metadata)}) 不匹配")
                print("   建议重新运行 'python utils/ingest_pdfs.py' 来重建索引")
            else:
                # 检查前几个metadata是否有文件信息
                sample_meta = vector_store.metadata[:3]
                has_file_info = any(
                    isinstance(m, dict) and isinstance(m.get('metadata'), dict) and m.get('metadata', {}).get('file')
                    for m in sample_meta
                )
                if not has_file_info:
                    print("⚠ 警告：metadata中缺少文件信息")
                    print("   建议重新运行 'python utils/ingest_pdfs.py' 来重建索引")
            
            # 尝试加载BM25索引
            bm25_retriever = None
            bm25_index_path = os.path.join(vectors_dir, 'bm25_index.pkl')
            if os.path.exists(bm25_index_path):
                try:
                    with open(bm25_index_path, 'rb') as f:
                        bm25_retriever = pickle.load(f)
                    print(f"✓ BM25索引已加载: {bm25_index_path}")
                except Exception as e:
                    print(f"⚠ 加载BM25索引失败: {e}，将仅使用向量检索")
            else:
                print("ℹ 未找到BM25索引，将仅使用向量检索（混合检索需要BM25索引）")
            
            # 初始化RAG检索器（支持混合检索）
            rag_retriever = RAGRetriever(vector_store, embedder, bm25_retriever=bm25_retriever)
        else:
            rag_retriever = None
            print("RAG未启用：未检测到已构建的向量数据，将跳过向量检索。")
            print("   请运行 'python utils/ingest_pdfs.py' 来构建向量索引")
    except Exception as e:
        rag_retriever = None
        print(f"RAG未启用：初始化检索器时发生异常: {e}")

    web_retriever = WebRetriever()
    reranker = Reranker()
    
    # 初始化对话记忆
    memory = ConversationMemory(max_history=10)
    memory.new_session()
    
    # 初始化智能体（传递RAG组件、web检索器、llm_client和memory）
    intent_router = IntentRouter()
    intake_wizard = IntakeWizard()
    contract_drafting_agent = ContractDraftingAgent(
        rag_retriever=rag_retriever, 
        reranker=reranker,
        llm_client=llm_client,
        memory=memory
    )
    contract_review_agent = ContractReviewAgent(
        rag_retriever=rag_retriever,
        reranker=reranker,
        llm_client=llm_client,
        memory=memory
    )
    legal_qa_agent = LegalQAAgent(
        rag_retriever=rag_retriever, 
        web_retriever=web_retriever,
        reranker=reranker,
        llm_client=llm_client,
        memory=memory
    )
    
    print("系统初始化完成！")


def build_vector_store(vector_store, embedder, data_loader):
    """构建向量数据库"""
    all_data = []
    all_metadata = []
    
    # 添加法律库
    for item in data_loader.get_legal_data():
        all_data.append(item.get('content', ''))
        all_metadata.append({
            'content': item.get('content', ''),
            'source': 'legal',
            'metadata': {
                'title': item.get('title', ''),
                'category': item.get('category', '')
            }
        })
    
    # 添加案例库
    for item in data_loader.get_case_data():
        all_data.append(item.get('content', ''))
        all_metadata.append({
            'content': item.get('content', ''),
            'source': 'case',
            'metadata': {
                'title': item.get('title', ''),
                'date': item.get('date', '')
            }
        })
    
    # 添加模板库
    for item in data_loader.get_template_data():
        all_data.append(item.get('content', ''))
        all_metadata.append({
            'content': item.get('content', ''),
            'source': 'template',
            'metadata': {
                'type': item.get('type', ''),
                'jurisdiction': item.get('jurisdiction', 'Singapore')
            }
        })
    
    if all_data:
        # 批量编码
        vectors = embedder.encode(all_data)
        # 添加到向量数据库
        vector_store.add(vectors, all_metadata)
        # 保存（记录实际使用的嵌入器模型名称）
        actual_model_name = getattr(embedder, '_actual_model_name', embedder.model_name if hasattr(embedder, 'model_name') else None)
        vector_store.save(embedder_model_name=actual_model_name)


def process_message(message: str, session_state) -> Tuple[str, str, dict]:
    """处理用户消息
    Returns:
        (response, chat_history_str, updated_session_state)
    """
    global memory, intent_router, intake_wizard, contract_drafting_agent, contract_review_agent, legal_qa_agent, rag_retriever, web_retriever, reranker
    # 若系统核心组件尚未初始化，进行惰性初始化
    if intent_router is None or legal_qa_agent is None:
        try:
            initialize_system()
        except Exception:
            # 最小可用降级，避免空引用
            if intent_router is None:
                intent_router = IntentRouter()
            if memory is None:
                memory = ConversationMemory(max_history=10)
                memory.new_session()
    # 懒加载：若系统尚未初始化或 memory 丢失，立即初始化一个会话内存
    if memory is None:
        memory = ConversationMemory(max_history=10)
        memory.new_session()
    
    # 确保 session_state 是字典类型（不要创建新字典，而是修改传入的字典）
    if session_state is None:
        session_state = {}
    elif not isinstance(session_state, dict):
        session_state = dict(session_state) if session_state else {}
    
    session_id = session_state.get('session_id')
    if not session_id:
        session_id = memory.new_session()
        session_state['session_id'] = session_id
    
    memory.set_session(session_id)
    
    # 添加用户消息到历史
    memory.add_message('user', message)
    
    response = ""
    
    # 检查是否正在执行合同起草任务（信息收集中）
    active_task = session_state.get('active_task')
    if active_task == 'contract_drafting':
        # 正在收集合同信息，优先继续收集流程，忽略意图识别
        contract_type = session_state.get('contract_type', 'msa_service')
        jurisdiction = session_state.get('jurisdiction', 'Singapore')
        
        # 检查用户是否明确要取消或切换任务
        cancel_keywords = ['取消', 'cancel', '停止', 'stop', '退出', 'exit', '不做了', '算了']
        if any(keyword in message.lower() for keyword in cancel_keywords):
            # 用户取消，清除任务状态
            session_state.pop('active_task', None)
            session_state.pop('contract_type', None)
            session_state.pop('jurisdiction', None)
            intake_wizard.clear_session(session_id)
            response = "Contract drafting cancelled. How can I help you?"
        else:
            # 检查用户消息是否是问句（包含疑问词）
            question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which', '什么', '如何', '为什么', '需要', 'needed', 'need', 'require', 'requirements']
            is_question = any(word in message.lower() for word in question_words)
            
            # 如果是问句，先回答问题，然后再继续收集信息
            if is_question and legal_qa_agent:
                # 先回答用户的问题
                qa_result = legal_qa_agent.answer(message, session_id=session_id)
                qa_answer = qa_result.get('answer', '')
                
                # 然后继续合同起草流程
                routing = {
                    'intent': 'contract_draft',
                    'contract_type': contract_type,
                    'jurisdiction': jurisdiction
                }
                contract_response = handle_contract_drafting(message, routing, session_id, session_state)
                
                # 合并回答和合同收集信息
                if qa_answer:
                    response = qa_answer + "\n\n--- Contract Information Collection ---\n" + contract_response
                else:
                    response = contract_response
            else:
                # 继续合同起草流程
                routing = {
                    'intent': 'contract_draft',
                    'contract_type': contract_type,
                    'jurisdiction': jurisdiction
                }
                response = handle_contract_drafting(message, routing, session_id, session_state)
    else:
        # 正常流程：进行意图识别
        routing = intent_router.route(message)
        intent = routing['intent']
        # print(f"[DEBUG process_message] Intent detected: {intent}")
        # print(f"[DEBUG process_message] Routing: {routing}")
        
        if intent == 'contract_draft':
            # 开始新的合同起草任务
            contract_type = routing.get('contract_type')
            if not contract_type:
                # 如果意图识别没有识别出合同类型，使用默认值
                contract_type = 'msa_service'
            # 确保状态被设置
            session_state['active_task'] = 'contract_drafting'
            session_state['contract_type'] = contract_type
            session_state['jurisdiction'] = routing.get('jurisdiction', 'Singapore')
            # 调试：打印状态信息
            # print(f"[DEBUG process_message] Setting active_task to contract_drafting, contract_type: {contract_type}")
            # print(f"[DEBUG process_message] session_state after setting: {session_state}")
            response = handle_contract_drafting(message, routing, session_id, session_state)
            # 调试：打印状态信息（调用后）
            # print(f"[DEBUG process_message] session_state after handle_contract_drafting: {session_state}")
            # print(f"[DEBUG process_message] active_task after handle: {session_state.get('active_task')}")
        elif intent == 'contract_review':
            response = handle_contract_review(message, routing, session_id)
        else:  # legal_qa
            response = handle_legal_qa(message, routing, session_id)
    
    # 添加助手回复到历史
    memory.add_message('assistant', response)
    
    # 确保 session_id 在 session_state 中
    session_state['session_id'] = session_id
    
    # # 调试：打印状态信息
    # print(f"[DEBUG process_message] session_state before return: {session_state}")
    # print(f"[DEBUG process_message] active_task: {session_state.get('active_task')}")
    # print(f"[DEBUG process_message] contract_type: {session_state.get('contract_type')}")
    # print(f"[DEBUG process_message] session_state keys: {list(session_state.keys())}")
    
    # 确保返回的字典包含所有字段（使用 copy 以确保状态正确传递）
    return_state = session_state.copy()
    # print(f"[DEBUG process_message] return_state keys: {list(return_state.keys())}")
    # print(f"[DEBUG process_message] return_state active_task: {return_state.get('active_task')}")
    
    return response, format_chat_history(session_id), return_state


def sanitize_markdown(text: str) -> str:
    """清洗文本以安全地在前端以Markdown展示，去除本地绝对路径等噪声。"""
    try:
        if not isinstance(text, str):
            text = str(text)
        # 隐藏项目本地绝对路径（例如 D:\Desktop\lanchain_LawAgent\ ...）
        project_root = os.path.abspath(os.path.dirname(__file__))
        # 规范化分隔符为反斜杠进行替换匹配
        project_root_windows = project_root.replace('/', '\\')
        # 移除精确项目路径前缀出现
        text = text.replace(project_root_windows + '\\', '')
        # 兜底：移除特定已知路径前缀（用户提出要隐藏）
        text = text.replace('D:\\Desktop\\lanchain_LawAgent\\', '')
        # 将字面量转义换行恢复为真实换行
        text = text.replace('\\r\\n', '\n').replace('\\n', '\n')
        # 适度清理多余空行（最多连续两行）
        while '\n\n\n' in text:
            text = text.replace('\n\n\n', '\n\n')
        # 防止被当作本地路径处理：若文本以盘符路径开头，前置一个不可见零宽空格
        # 但仅当整段开头疑似Windows盘符时
        if len(text) > 2 and text[1] == ':' and (text[0].isalpha()):
            text = '\u200b' + text
        return text
    except Exception:
        return str(text)


def handle_contract_drafting(message: str, routing: Dict, session_id: str, session_state: dict = None) -> str:
    """处理合同起草"""
    global intake_wizard, contract_drafting_agent, memory, legal_qa_agent
    
    if session_state is None:
        session_state = {}
    
    # 如果是问句（包含what, how, why等疑问词），先回答问题
    question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which', '什么', '如何', '为什么']
    is_question = any(word in message.lower() for word in question_words)
    
    response_parts = []
    
    # 如果是问句，先回答问题
    if is_question and legal_qa_agent:
        qa_result = legal_qa_agent.answer(message, session_id=session_id)
        answer = qa_result.get('answer', '')
        if answer:
            response_parts.append(answer)
            response_parts.append("\n\nIf you would like to draft a contract, I will need some additional information:")
    
    # 获取当前会话的合同规格（如果存在）
    contract_type = routing.get('contract_type', 'msa_service')
    existing_spec = None
    if session_id:
        existing_spec = intake_wizard.get_session_state(session_id)
    
    # 收集信息
    session_context = {
        'session_id': session_id,
    }
    if existing_spec:
        session_context['contract_spec'] = existing_spec
    
    collection_result = intake_wizard.collect(
        message,
        contract_type,
        session_context=session_context,
    )
    
    # 检查用户是否明确表示可以开始起草（即使缺少一些字段）
    proceed_keywords = ['start to draft', 'okay', 'start', 'proceed', 'continue', 'draft', 'generate', 'go ahead', 'yes', 'ok', 'okay']
    user_wants_to_proceed = any(keyword in message.lower() for keyword in proceed_keywords)
    
    # 检查核心字段是否已收集
    core_complete = collection_result.get('core_complete', False)
    
    # 如果核心字段已收集且用户明确表示可以开始，或者所有字段都已收集，则生成合同
    if (core_complete and user_wants_to_proceed) or collection_result['is_complete']:
        # 信息完整或核心字段已收集且用户同意，生成合同
        contract_spec = collection_result['contract_spec']
        contract = contract_drafting_agent.draft(
            contract_spec,
            contract_type,
            routing.get('jurisdiction', 'Singapore')
        )
        
        # 合同生成完成，清除任务状态
        session_state.pop('active_task', None)
        session_state.pop('contract_type', None)
        session_state.pop('jurisdiction', None)
        intake_wizard.clear_session(session_id)
        
        contract_msg = f"Contract drafted:\\n\\n```\n{contract['content']}\n```\\n\\nPending items: {len(contract['todo_placeholders'])}"
        
        response_parts.append(contract_msg)
        return sanitize_markdown("\n".join(response_parts))
    
    # 信息未完整，继续收集
    if not collection_result['is_complete']:
        # 确保状态被设置（即使已经设置过也要确保）
        if session_state is not None:
            session_state['active_task'] = 'contract_drafting'
            session_state['contract_type'] = contract_type
            session_state['jurisdiction'] = routing.get('jurisdiction', 'Singapore')
        
        # 检查是否提取到了新信息
        extracted_info = collection_result.get('contract_spec', {})
        previous_spec = existing_spec or {}
        
        # 检查是否有新信息被提取
        has_new_info = False
        newly_collected_fields = []
        for key, value in extracted_info.items():
            if key not in previous_spec or previous_spec[key] != value:
                has_new_info = True
                newly_collected_fields.append(key)
        
        # 如果有新信息被提取，先确认已收集的信息
        if has_new_info and newly_collected_fields:
            confirmation_parts = []
            for field in newly_collected_fields:
                value = extracted_info.get(field)
                if value:
                    if field == 'parties' and isinstance(value, list):
                        confirmation_parts.append(f"✓ Parties: {', '.join(value)}")
                    elif field == 'services':
                        confirmation_parts.append(f"✓ Services: {value}")
                    elif field == 'duration':
                        confirmation_parts.append(f"✓ Duration: {value}")
                    elif field == 'payment_terms':
                        confirmation_parts.append(f"✓ Payment Terms: {value}")
                    else:
                        # 格式化字段名（将下划线替换为空格并首字母大写）
                        field_display = field.replace('_', ' ').title()
                        if isinstance(value, list):
                            confirmation_parts.append(f"✓ {field_display}: {', '.join(str(v) for v in value)}")
                        else:
                            confirmation_parts.append(f"✓ {field_display}: {value}")
            
            if confirmation_parts:
                response_parts.append("I've collected the following information:")
                response_parts.append("\n".join(confirmation_parts))
        
        # 如果核心字段已收集，询问用户是否使用默认值继续
        if core_complete and not user_wants_to_proceed:
            missing_fields = collection_result.get('missing_fields', [])
            if missing_fields:
                response_parts.append(f"\n\nI have the core information (parties and services). The following fields are still missing: {', '.join(missing_fields)}")
                response_parts.append("\nI can draft the contract now using default values for the missing fields, or you can provide more details.")
                response_parts.append("\nWould you like me to proceed with drafting? (You can say 'yes', 'proceed', 'continue', 'start', etc.)")
        # 如果用户消息是问句且已经回答了问题，或者没有提取到新信息，给出更友好的提示
        elif is_question and not has_new_info:
            # 用户问了问题，但没有提供新信息，只显示收集信息提示（不重复问题）
            nq = collection_result['next_question']
            if nq:
                # 如果已经有回答，只显示收集信息提示
                if response_parts:
                    response_parts.append(f"\n\nTo continue drafting the contract, please provide: {nq}")
                else:
                    response_parts.append(f"To continue drafting the contract, please provide: {nq}")
            else:
                missing_fields = collection_result.get('missing_fields', [])
                if missing_fields:
                    response_parts.append(f"\n\nTo continue drafting the contract, please provide the following information: {', '.join(missing_fields)}")
                else:
                    response_parts.append("\n\nTo continue drafting the contract, please provide the required information.")
        else:
            # 正常流程：显示下一个问题
            nq = collection_result['next_question']
            if nq is None:
                missing_fields = collection_result.get('missing_fields', [])
                if missing_fields:
                    if response_parts:
                        response_parts.append(f"\n\nPlease provide the following information: {', '.join(missing_fields)}")
                    else:
                        response_parts.append(f"Please provide the following information: {', '.join(missing_fields)}")
                else:
                    if response_parts:
                        response_parts.append("\n\nPlease provide the required information.")
                    else:
                        response_parts.append("Please provide the required information.")
            else:
                if response_parts:
                    response_parts.append(f"\n\n{nq}")
                else:
                    response_parts.append(str(nq))
        
        return sanitize_markdown("\n".join(response_parts))


def handle_contract_review(message: str, routing: Dict, session_id: str) -> str:
    """处理合同审查"""
    global contract_review_agent, memory
    
    # 从上下文中获取合同内容
    context = memory.get_context(session_id)
    
    # 创建示例合同（实际中应从message中提取）
    contract = {
        'content': context if 'contract' in context.lower() else message,
        'type': routing.get('contract_type', 'unknown')
    }
    
    # 审阅
    review = contract_review_agent.review(contract)
    
    return sanitize_markdown(format_review_report(review))


def handle_legal_qa(message: str, routing: Dict, session_id: str) -> str:
    """处理法律咨询"""
    global legal_qa_agent
    
    # Answer the question, pass session_id to get context from memory, force English
    result = legal_qa_agent.answer(message, session_id=session_id)
    return sanitize_markdown(format_qa_result(result))


def format_qa_result(result: Dict) -> str:
    """格式化QA结果"""
    answer = result['answer']
    
    response_parts = [answer]
    # 在答案后统一展示一次置信度，避免与证据段落冲突
    if result.get('confidence'):
        response_parts.append(f"\n\nConfidence: {result['confidence']}")
    
    # 只有当确实有检索到证据时才显示证据部分（改为统一列表并去重）
    if result['has_evidence']:
        response_parts.append("\n\n--- Evidence Retrieved ---")

        seen = set()
        unique_items = []
        web_evidence_items = []  # 单独存储Web证据以便添加URL
        
        # 处理主要证据（RAG检索结果）
        for ev in (result.get('primary_evidence') or [])[:10]:
            meta = ev.get('metadata', {}) or {}
            
            # 调试：检查metadata内容
            if not meta:
                print(f"[DEBUG format_qa_result] 警告：证据metadata为空，ev keys: {list(ev.keys())}")
                # 尝试从content中提取一些信息
                content = ev.get('content', '')[:100] if ev.get('content') else ''
                if content:
                    # 尝试从content中提取可能的标题或关键词
                    first_line = content.split('\n')[0][:50] if content else ''
                    if first_line:
                        meta = {'title': first_line, 'file': 'Unknown', 'category': 'Unknown'}
            
            file_full = meta.get('file', '')
            category = meta.get('category', '')
            title = meta.get('title', '')
            source = ev.get('source', 'rag') or 'rag'
            
            # 构建显示项：优先使用文件名，如果没有则使用标题
            display_parts = []
            
            # 添加类别（如果有且不是Unknown）
            if category and category != 'Unknown':
                display_parts.append(category)
            
            # 添加文件名或标题
            if file_full and file_full != 'Unknown':
                # 提取文件名（去除路径）
                file_name = file_full.replace('\\', '/').split('/')[-1]
                if file_name:
                    display_parts.append(file_name)
            
            if title and title not in display_parts and title != 'Unknown':
                display_parts.append(title)
            
            # 如果都没有，至少显示source和内容摘要
            if not display_parts:
                content_preview = ev.get('content', '')[:50] if ev.get('content') else ''
                if content_preview:
                    display_parts.append(f"Document: {content_preview}...")
                else:
                    display_parts.append(f"Source: {source}")
            
            # 组合显示项
            key = ' / '.join(display_parts) if display_parts else f"Source: {source}"
            
            # 使用完整的标识（包括类别和文件名）作为去重键
            unique_key = f"rag_{category}_{file_full}_{title}_{ev.get('content', '')[:50]}" if file_full or title else f"rag_{source}_{ev.get('content', '')[:50]}"
            if unique_key not in seen:
                seen.add(unique_key)
                unique_items.append(key)
        
        # 处理次要证据（Web检索结果），添加URL
        for ev in (result.get('secondary_evidence') or [])[:10]:
            meta = ev.get('metadata', {}) or {}
            title = meta.get('title', '')
            url = meta.get('url', '')
            source = ev.get('source', 'web')
            
            # 构建显示项
            if title:
                display_key = title
            else:
                display_key = source
            
            # 去重处理：使用URL作为唯一标识（如果有）
            if url:
                unique_key = f"web_{url}"
            else:
                unique_key = f"web_{title}_{source}" if title else f"web_{source}"
            
            if unique_key not in seen:
                seen.add(unique_key)
                
                # 如果有URL，添加链接
                if url:
                    # 使用Markdown格式的链接
                    display_item = f"{display_key} - [View Original Link]({url})"
                    web_evidence_items.append(display_item)
                else:
                    web_evidence_items.append(display_key)

        # 先显示RAG证据
        if unique_items:
            response_parts.append("\n\n**RAG Evidence:**")
            for item in unique_items[:6]:
                response_parts.append(f"\n- {item}")
        else:
            # 如果没有文件信息，至少显示有证据
            if result.get('primary_evidence'):
                response_parts.append(f"\n\n**RAG Evidence:** {len(result.get('primary_evidence', []))} document(s) retrieved")
        
        # 再显示Web证据（带URL）
        if web_evidence_items:
            response_parts.append("\n\n**Web Evidence:**")
            for item in web_evidence_items[:6]:
                response_parts.append(f"\n- {item}")

        if result.get('uncertainty_note'):
            response_parts.append(f"\n\n{result['uncertainty_note']}")
    
    return "".join(response_parts)


def format_review_report(review: Dict) -> str:
    """格式化审阅报告"""
    parts = [
        f"Contract Review Report\\n",
        f"Severity: {review['severity']}\\n",
        f"Confidence: {review['confidence']:.2f}\\n\\n"
    ]
    
    if review['risk_points']:
        parts.append("⚠️ Risk Points:\\n")
        for risk in review['risk_points'][:5]:
            parts.append(f"- [{risk['severity']}] {risk['description']}\\n")
            parts.append(f"  Recommendation: {risk.get('recommendation', '')}\\n")
        parts.append("\\n")
    
    if review['recommendations']:
        parts.append("💡 Recommendations:\\n")
        for rec in review['recommendations'][:5]:
            parts.append(f"- [{rec['priority']}] {rec['description']}\\n")
    
    return "".join(parts)


def format_chat_history(session_id: str) -> str:
    """格式化聊天历史"""
    global memory
    
    history = memory.get_history(session_id)
    
    lines = []
    for msg in history[-5:]:  # Show only last 5 messages
        role = "User" if msg['role'] == 'user' else "Assistant"
        # Truncate long messages (show only first 100 characters)
        content = msg['content']
        if len(content) > 100:
            content = content[:100] + "..."
        lines.append(f"{role}: {content}")
    
    return "\\n---\\n".join(lines)


# Gradio界面
def create_interface():
    """创建Gradio界面"""
    
    with gr.Blocks(title="Intelligent Legal Advice Assistants in Singapore", theme=gr.themes.Soft(), css='''
    /* 为整体界面设置中英文字体：优先 Times New Roman，其次宋体 */
    .gradio-container, body, .gr-block, .gr-textbox, .gr-chatbot, .gr-markdown, .gr-button {
        font-family: "Times New Roman", "SimSun", serif;
    }
    /* Chatbot 气泡内文本 */
    .wrap.svelte-1n6ueqj, .message.svelte-1n6ueqj, .bot.svelte-1n6ueqj, .user.svelte-1n6ueqj {
        font-family: "Times New Roman", "SimSun", serif;
        white-space: pre-wrap;
    }
    ''') as demo:
        gr.Markdown(
            "# Singapore's Intelligent Legal Advice Assistants\n"
            "Provide legal advice services based on retrieval-augmented generation, supporting contract drafting, review and legal consultation."
        )
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Chat Window",
                    height=1000,
                    # avatar_images=("🤖", "👤"),
                    avatar_images=("img/用户.png","img/机器人.png" ), # 自定义头像
                    render_markdown=True
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        label="Input Question",
                        placeholder="Please input your question or demand...",
                        lines=1,
                        scale=4
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1)
                
                with gr.Row():
                    clear_btn = gr.Button("Clear Conversation")
                    new_session_btn = gr.Button("New Session")
            
            with gr.Column(scale=2):
                gr.Markdown("### Session Information")
                session_info = gr.Textbox(
                    label="Current Session",
                    interactive=False,
                    lines=2,
                    scale=None,
                )
                
                gr.Markdown("### Tips: ")
                gr.Markdown(
                    """
                    - **contract_drafting**：input“contract_drafting”，, and give the names of both parties to the contract. For example: help me formulate a service contract, Party A: XXX, Party B: XXX.
                    - **contract_review**：input“contract_review”，and give the content of the contract. For example: help me review this contract, contract content: XXX.
                    - **legal_qa**：indirectly ask the question. For example: I would like to know about the laws related to disputes between companies and business entities in Singapore.
                    """,
                    elem_id=None,
                    visible=True
                )
        
        # 使用 State 来保持会话状态
        session_state = gr.State(value={})
        
        # 事件处理
        def respond(message, history, state):
            if not message:
                return history, "", state, ""
            
            # 如果 state 为空，初始化为空字典
            if state is None:
                state = {}
            
            result = process_message(message, state)
            # process_message 返回 (response, chat_history_str, updated_session_state)
            # print(f"[DEBUG respond] result type: {type(result)}, result length: {len(result) if isinstance(result, tuple) else 'N/A'}")
            if isinstance(result, tuple) and len(result) >= 3:
                response_text = result[0]
                session_history = result[1]
                updated_state = result[2]  # 获取更新后的状态
                # print(f"[DEBUG respond] Got updated_state from result[2]: {updated_state}")
            elif isinstance(result, tuple) and len(result) >= 2:
                response_text = result[0]
                session_history = result[1]
                updated_state = state  # 如果没有返回状态，使用原状态
                # print(f"[DEBUG respond] Using original state: {updated_state}")
            elif isinstance(result, tuple) and len(result) >= 1:
                response_text = result[0]
                session_history = ""
                updated_state = state
                # print(f"[DEBUG respond] Using original state (fallback): {updated_state}")
            else:
                response_text = result
                session_history = ""
                updated_state = state
                # print(f"[DEBUG respond] Using original state (no tuple): {updated_state}")
            
            # Ensure frontend displays as Markdown and sanitize potential local paths and escapes
            response_text = sanitize_markdown(response_text)
            history.append([message, response_text])
            
            # Format session information display - 使用更新后的状态
            session_id = updated_state.get('session_id', 'Unknown')
            session_info_parts = [f"Session ID: {session_id}"]
            
            # # 调试：打印状态信息
            # print(f"[DEBUG respond] updated_state keys: {list(updated_state.keys())}")
            # print(f"[DEBUG respond] updated_state: {updated_state}")
            
            # 显示任务状态
            active_task = updated_state.get('active_task')
            # print(f"[DEBUG respond] active_task: {active_task}")
            if active_task:
                session_info_parts.append(f"\nActive Task: {active_task}")
                if active_task == 'contract_drafting':
                    contract_type = updated_state.get('contract_type', 'N/A')
                    jurisdiction = updated_state.get('jurisdiction', 'N/A')
                    session_info_parts.append(f"Contract Type: {contract_type}")
                    session_info_parts.append(f"Jurisdiction: {jurisdiction}")
            
            # 显示当前会话收集的信息（current_session）
            global intake_wizard
            # print(f"[DEBUG respond] intake_wizard: {intake_wizard is not None}, session_id: {session_id}")
            if intake_wizard and session_id and session_id != 'Unknown':
                current_session = intake_wizard.get_session_state(session_id)
                # print(f"[DEBUG respond] current_session: {current_session}")
                if current_session:
                    session_info_parts.append("\n--- Current Session (Collected Info) ---")
                    for key, value in current_session.items():
                        if value:  # 只显示有值的字段
                            # 格式化显示
                            if isinstance(value, list):
                                value_str = ', '.join(str(v) for v in value)
                            else:
                                value_str = str(value)
                            # 限制显示长度
                            if len(value_str) > 50:
                                value_str = value_str[:50] + "..."
                            session_info_parts.append(f"  {key}: {value_str}")
            
            # 显示最近历史
            if session_history:
                session_info_parts.append(f"\n--- Recent History ---\n{session_history}")
            
            session_info_text = "\n".join(session_info_parts)
            
            return history, "", updated_state, session_info_text
        
        def clear(state):
            global intake_wizard
            # 清除当前会话的任务状态
            if state and state.get('session_id'):
                old_session_id = state.get('session_id')
                if intake_wizard:
                    intake_wizard.clear_session(old_session_id)
            # 返回空状态和空的session_info
            cleared_state = {}
            session_info_text = "Session cleared."
            return "", [], cleared_state, session_info_text
        
        def new_session(state):
            global memory, intake_wizard
            if memory:
                # 清除旧会话的任务状态
                if state and state.get('session_id'):
                    old_session_id = state.get('session_id')
                    if intake_wizard:
                        intake_wizard.clear_session(old_session_id)
                
                # 创建新会话
                session_id = memory.new_session()
                new_state = {
                    "session_id": session_id
                }
                session_info_text = f"Session ID: {session_id}\n\nNew session created."
                return new_state, session_info_text
            return {}, "No session available."
        
        submit_btn.click(
            respond,
            inputs=[msg, chatbot, session_state],
            outputs=[chatbot, msg, session_state, session_info]
        )
        
        msg.submit(
            respond,
            inputs=[msg, chatbot, session_state],
            outputs=[chatbot, msg, session_state, session_info]
        )
        
        clear_btn.click(clear, inputs=[session_state], outputs=[msg, chatbot, session_state, session_info])
        
        new_session_btn.click(
            new_session,
            inputs=[session_state],
            outputs=[session_state, session_info]
        )
        
        demo.load(initialize_system)
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860, inbrowser=True)

