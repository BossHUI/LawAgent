"""
Conversation Memory - 对话记忆系统
"""
from typing import List, Dict, Optional
from collections import deque
from datetime import datetime


class ConversationMemory:
    """对话记忆系统 - 支持多轮对话"""
    
    def __init__(self, max_history: int = 10):
        """
        初始化对话记忆
        
        Args:
            max_history: 最大历史记录数
        """
        self.max_history = max_history
        self.conversations = {}  # session_id -> conversation history
        self.current_session_id = None
    
    def new_session(self) -> str:
        """创建新会话"""
        session_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        self.conversations[session_id] = {
            'messages': deque(maxlen=self.max_history),
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'message_count': 0
            }
        }
        self.current_session_id = session_id
        return session_id
    
    def set_session(self, session_id: str):
        """设置当前会话"""
        if session_id not in self.conversations:
            self.conversations[session_id] = {
                'messages': deque(maxlen=self.max_history),
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'message_count': 0
                }
            }
        self.current_session_id = session_id
    
    def add_message(self, role: str, content: str, session_id: Optional[str] = None):
        """
        添加消息
        
        Args:
            role: 角色 ('user' 或 'assistant')
            content: 消息内容
            session_id: 会话ID（可选，默认使用当前会话）
        """
        if session_id is None:
            session_id = self.current_session_id
        
        if session_id and session_id in self.conversations:
            message = {
                'role': role,
                'content': content,
                'timestamp': datetime.now().isoformat()
            }
            self.conversations[session_id]['messages'].append(message)
            self.conversations[session_id]['metadata']['message_count'] += 1
        else:
            raise ValueError(f"会话 {session_id} 不存在")
    
    def get_history(self, session_id: Optional[str] = None, n: Optional[int] = None) -> List[Dict]:
        """
        获取历史记录
        
        Args:
            session_id: 会话ID（可选，默认使用当前会话）
            n: 返回最近n条记录（None表示返回所有记录）
            
        Returns:
            历史记录列表（按时间顺序）
        """
        if session_id is None:
            session_id = self.current_session_id
        
        if session_id and session_id in self.conversations:
            messages = list(self.conversations[session_id]['messages'])
            if n and n > 0:
                return messages[-n:]
            return messages
        return []
    
    def get_context(self, session_id: Optional[str] = None, max_messages: int = None) -> str:
        """
        获取上下文字符串
        
        Args:
            session_id: 会话ID
            max_messages: 最大消息数量（可选，用于限制上下文长度）
            
        Returns:
            上下文字符串
        """
        history = self.get_history(session_id)
        
        # 如果指定了最大消息数，只取最近的消息
        if max_messages:
            history = history[-max_messages:]
        
        context_parts = []
        
        for msg in history:
            role = msg['role']
            content = msg['content']
            context_parts.append(f"{role}: {content}")
        
        return "\n".join(context_parts)
    
    def clear_session(self, session_id: Optional[str] = None):
        """清空会话"""
        if session_id is None:
            session_id = self.current_session_id
        
        if session_id and session_id in self.conversations:
            self.conversations[session_id]['messages'].clear()
            self.conversations[session_id]['metadata']['message_count'] = 0
    
    def get_statistics(self, session_id: Optional[str] = None) -> Dict:
        """获取统计信息"""
        if session_id is None:
            session_id = self.current_session_id
        
        if session_id and session_id in self.conversations:
            return self.conversations[session_id]['metadata'].copy()
        return {}
    
    def list_sessions(self) -> List[Dict]:
        """列出所有会话"""
        sessions = []
        for session_id, data in self.conversations.items():
            sessions.append({
                'session_id': session_id,
                'metadata': data['metadata'].copy(),
                'message_count': len(data['messages'])
            })
        return sessions

