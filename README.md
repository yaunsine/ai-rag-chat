# FastAPI + Jinja 文档检索问答系统

基于FastAPI和Jinja模板构建的知识库问答系统，提供智能文档检索和问答功能。

## 项目结构

```
keda_llm/
├── fastapi_server.py          # FastAPI后端服务器
├── flask_server.py            # Flask后端服务器（旧版）
├── run_fastapi.py             # FastAPI启动脚本
├── run_flask.py               # Flask启动脚本（旧版）
├── config.py                  # 系统配置文件
├── requirements.txt           # Python依赖包
├── templates/                 # Jinja模板目录
│   ├── base.html             # 基础模板
│   ├── index.html            # 标准版界面模板
│   └── simple_index.html     # 简化版界面模板
├── static/                    # 静态文件目录
└── README_JINJA.md           # 项目说明文档
```

## 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 启动后端服务
确保后端服务在 `http://localhost:8000` 运行

### 启动前端服务
```bash
# 使用FastAPI版本（推荐）
python run_fastapi.py
```

访问以下地址使用系统：
- **简化版界面**（默认）：`http://localhost:5000` - 不显示IP输入框，不调节返回文档数量，对话区域只有一个输入框
- **标准版界面**：`http://localhost:5000/standard` - 完整功能界面
- **纯简化版**：`http://localhost:5000/simple` - 直接访问简化版界面

### 界面配置
在 `config.py` 中可以修改前端界面配置：
```python
# 前端界面配置
SIMPLIFIED_UI = True  # 是否使用简化版界面
SHOW_IP_INPUT = False  # 是否显示IP输入框
SHOW_PARAM_CONTROLS = False  # 是否显示参数调节控件
SINGLE_INPUT_MODE = True  # 是否使用单一输入框模式
```

## 功能特性

### 简化版界面特性
- 🚫 **无IP输入框** - 后端API地址在配置中预设
- 🚫 **无参数调节控件** - 使用默认查询参数
- 💬 **单一输入框** - 对话区域只有一个输入框，简化操作
- 📱 **响应式设计** - 适配不同屏幕尺寸
- 🎯 **专注对话** - 减少干扰，专注于问答体验

### 核心功能
- 📚 **文档加载** - 支持文件上传和文档加载
- 💬 **智能问答** - 基于RAG技术的智能问答
- 🗂️ **会话管理** - 多会话支持，独立对话历史
- 🔍 **文档检索** - 智能检索相关文档片段
- 🧠 **大模型集成** - 结合大语言模型的回答生成

## API接口

### FastAPI版本接口
- `GET /` - 主页面（根据配置显示简化版或标准版）
- `GET /simple` - 简化版主页面
- `GET /standard` - 标准版主页面
- `POST /api/sessions` - 创建会话
- `DELETE /api/sessions/{session_id}` - 删除会话
- `POST /api/upload_and_load` - 上传并加载文档
- `POST /api/query` - 提交查询
- `GET /api/status` - 获取会话状态
- `GET /api/health` - 健康检查

### 后端代理接口
所有API请求都会代理到后端服务 `http://localhost:8000`

## 部署指南

### 开发环境
1. 安装Python 3.8+
2. 安装依赖：`pip install -r requirements.txt`
3. 启动后端服务
4. 启动前端服务：`python run_fastapi.py`
5. 访问 `http://localhost:5000`

### 生产环境
- 使用uvicorn部署FastAPI应用
- 配置反向代理（如Nginx）
- 设置环境变量
- 配置SSL证书

## 技术栈

### 前端
- **FastAPI** - 现代Python Web框架
- **Jinja2** - 模板引擎
- **Bootstrap 5** - 响应式UI框架
- **Font Awesome** - 图标库

### 后端
- **RAG技术** - 检索增强生成
- **向量数据库** - 文档向量化存储
- **大语言模型** - 回答生成

### 配置管理
- **config.py** - 集中式配置管理
- 环境变量支持
- 默认值配置

## 迁移说明

### 从Flask迁移到FastAPI
项目已从Flask迁移到FastAPI，主要优势：
- ⚡ **更高性能** - 基于ASGI的异步处理
- 🔒 **类型安全** - 内置数据验证和序列化
- 📚 **自动文档** - OpenAPI自动生成
- 🛠️ **现代架构** - 依赖注入、中间件支持

### 保留功能
- Flask版本仍保留在 `flask_server.py`
- 所有API接口保持兼容
- 模板和静态文件复用

## 界面对比

### 简化版界面
- 隐藏IP地址输入框
- 移除参数调节滑块
- 单一输入框设计
- 简化侧边栏功能
- 专注核心问答体验

### 标准版界面
- 完整的参数调节功能
- API地址自定义
- 多输入框支持
- 完整的功能面板
- 适合高级用户使用

## 故障排除

### 故障排除

#### 常见问题
1. **"请先创建会话"错误**：
   - 简化版界面会在页面加载时自动创建会话
   - 如果失败，请检查后端服务是否正常运行
   - 点击左侧"创建新会话"按钮手动重试

2. **后端连接失败**：
   - 确保后端服务运行在 `http://localhost:8000`
   - 检查防火墙设置是否允许连接
   - 验证网络连接是否正常

3. **端口占用**：修改 `config.py` 中的端口配置
4. **依赖安装失败**：使用Python 3.8+版本

#### 简化版界面改进
- ✅ **自动会话创建** - 页面加载时自动创建新会话
- ✅ **智能错误处理** - 根据错误类型显示详细的故障排除指南
- ✅ **输入框状态管理** - 会话创建成功后才启用输入框
- ✅ **重试机制** - 提供一键重试创建会话功能

#### 日志查看
启动脚本会自动显示连接状态和错误信息