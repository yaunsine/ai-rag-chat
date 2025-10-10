# 文档检索问答系统 - Flask + Jinja 版本

## 概述

本项目已将原有的Streamlit应用转换为使用Flask + Jinja模板的Web应用，提供了更灵活的前端定制能力。

## 项目结构

```
keda_llm/
├── templates/           # Jinja模板目录
│   ├── base.html       # 基础模板
│   └── index.html      # 主页面模板
├── flask_server.py      # Flask后端服务器
├── run_flask.py        # 启动脚本
├── requirements.txt    # 依赖文件（已更新）
└── README_JINJA.md     # 本说明文件
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动后端API服务

确保原有的后端API服务正在运行：

```bash
# 假设后端服务在 server.py 中
python server.py
```

### 3. 启动Flask前端

```bash
# 方式1：使用启动脚本
python run_flask.py

# 方式2：直接运行Flask应用
python flask_server.py
```

### 4. 访问应用

打开浏览器访问：http://localhost:5000

## 功能特性

### ✅ 已实现功能

- **会话管理**: 创建、删除会话
- **文档加载**: 支持手动输入、文件上传、文件夹加载
- **智能问答**: 基于RAG技术的文档检索和回答
- **响应式设计**: 使用Bootstrap 5的现代化界面
- **实时交互**: AJAX请求，无需页面刷新

### 🎨 界面特色

- **现代化设计**: 使用Bootstrap 5框架
- **响应式布局**: 适配桌面和移动设备
- **中文优化**: 完整的中文界面和提示
- **交互友好**: 实时反馈和状态提示

## API接口

Flask前端代理了所有后端API调用：

| 端点 | 方法 | 描述 |
|------|------|------|
| `/` | GET | 主页面 |
| `/api/sessions` | POST | 创建会话 |
| `/api/sessions/<id>` | DELETE | 删除会话 |
| `/api/load_documents` | POST | 加载文档 |
| `/api/upload_and_load` | POST | 上传文件 |
| `/api/load_folder` | POST | 加载文件夹 |
| `/api/query` | POST | 查询问题 |
| `/api/status` | GET | 获取状态 |
| `/health` | GET | 健康检查 |

## 配置选项

### 环境变量

```bash
# Flask服务器配置
FLASK_HOST=0.0.0.0      # 监听地址
FLASK_PORT=5000         # 监听端口
FLASK_DEBUG=True        # 调试模式

# 后端API地址
API_BASE_URL=http://localhost:8000
```

### 自定义配置

可以在 `flask_server.py` 中修改以下配置：

```python
# 后端API地址
API_BASE_URL = "http://localhost:8000"

# 静态文件配置（如果需要）
app = Flask(__name__, static_folder='static', template_folder='templates')
```

## 开发指南

### 修改模板

模板文件位于 `templates/` 目录：

- `base.html`: 基础模板，包含HTML结构和CSS/JS引用
- `index.html`: 主页面模板，包含所有功能界面

### 添加新功能

1. **修改模板**: 在 `index.html` 中添加新的HTML元素
2. **添加JavaScript**: 在模板的 `{% block scripts %}` 部分添加交互逻辑
3. **添加API端点**: 在 `flask_server.py` 中添加新的路由处理

### 样式定制

项目使用Bootstrap 5，可以通过以下方式定制样式：

1. **修改CSS**: 在 `base.html` 的 `<style>` 标签中添加自定义CSS
2. **使用Bootstrap类**: 直接使用Bootstrap的CSS类
3. **添加自定义CSS文件**: 在 `static/` 目录添加CSS文件并在模板中引用

## 部署说明

### 生产环境部署

1. **禁用调试模式**:
   ```bash
   export FLASK_DEBUG=False
   ```

2. **使用生产服务器**:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 flask_server:app
   ```

3. **配置反向代理** (可选):
   ```nginx
   location / {
       proxy_pass http://localhost:5000;
       proxy_set_header Host $host;
       proxy_set_header X-Real-IP $remote_addr;
   }
   ```

## 故障排除

### 常见问题

1. **端口占用**: 修改 `FLASK_PORT` 环境变量或使用其他端口
2. **API连接失败**: 检查后端服务是否运行在 `http://localhost:8000`
3. **依赖问题**: 重新安装依赖 `pip install -r requirements.txt`

### 日志查看

Flask服务器会在控制台输出访问日志和错误信息。

## 从Streamlit迁移的优势

1. **更灵活的界面定制**: 完全控制HTML/CSS/JS
2. **更好的性能**: 静态文件缓存，减少重复加载
3. **更标准的Web开发**: 使用行业标准的模板引擎
4. **更容易扩展**: 可以轻松添加新的页面和功能

## 技术支持

如有问题，请检查：
- 后端API服务是否正常运行
- 所有依赖是否已正确安装
- 防火墙设置是否允许端口访问

---
**文档检索问答系统** | 基于RAG技术构建 | Flask + Jinja版本