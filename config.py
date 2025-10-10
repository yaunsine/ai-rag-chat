"""
配置文件 - 管理后端API地址和系统参数
"""

import os
from typing import Dict, Any

class Config:
    """系统配置类"""
    
    # 后端API地址 - 从环境变量读取或使用默认值
    API_BASE_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")
    
    # 默认查询参数
    DEFAULT_TOP_K = 5
    DEFAULT_SIMILARITY_THRESHOLD = 0.7
    
    # 前端服务器配置
    FRONTEND_HOST = os.getenv("FRONTEND_HOST", "0.0.0.0")
    FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", "5000"))
    DEBUG_MODE = True
    
    # 静态文件配置
    STATIC_FILES_DIR = "static"
    TEMPLATES_DIR = "templates"
    
    # 会话配置
    SESSION_TIMEOUT = 3600  # 会话超时时间（秒）
    MAX_SESSIONS_PER_USER = 10
    
    # 前端界面配置
    SIMPLIFIED_UI = True  # 是否使用简化版界面
    SHOW_IP_INPUT = False  # 是否显示IP输入框
    SHOW_PARAM_CONTROLS = False  # 是否显示参数调节控件
    SINGLE_INPUT_MODE = True  # 是否使用单一输入框模式
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """获取完整配置信息"""
        return {
            "api_base_url": cls.API_BASE_URL,
            "default_top_k": cls.DEFAULT_TOP_K,
            "default_similarity_threshold": cls.DEFAULT_SIMILARITY_THRESHOLD,
            "frontend_host": cls.FRONTEND_HOST,
            "frontend_port": cls.FRONTEND_PORT,
            "static_files_dir": cls.STATIC_FILES_DIR,
            "templates_dir": cls.TEMPLATES_DIR,
            "session_timeout": cls.SESSION_TIMEOUT
        }
    
    @classmethod
    def validate_config(cls) -> bool:
        """验证配置是否有效"""
        try:
            # 检查必要的配置项
            if not cls.API_BASE_URL:
                raise ValueError("API_BASE_URL不能为空")
            
            if cls.DEFAULT_TOP_K <= 0:
                raise ValueError("DEFAULT_TOP_K必须大于0")
            
            if not 0 <= cls.DEFAULT_SIMILARITY_THRESHOLD <= 1:
                raise ValueError("DEFAULT_SIMILARITY_THRESHOLD必须在0-1之间")
            
            return True
        except Exception as e:
            print(f"配置验证失败: {e}")
            return False

# 创建配置实例
config = Config()

if __name__ == "__main__":
    # 测试配置
    print("当前配置:")
    for key, value in config.get_config().items():
        print(f"{key}: {value}")
    
    print(f"\n配置验证: {'通过' if config.validate_config() else '失败'}")