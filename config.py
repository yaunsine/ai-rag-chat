"""
应用配置加载模块
从 config.yaml 读取配置，支持环境变量覆盖敏感字段
"""

import os
import yaml
from pathlib import Path
from typing import Any


_config_cache = None


def _find_config() -> Path:
    """查找配置文件，优先使用环境变量指定的路径"""
    env_path = os.environ.get("APP_CONFIG")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path

    default = Path(__file__).parent / "config.yaml"
    if default.exists():
        return default

    raise FileNotFoundError(
        "未找到配置文件 config.yaml。"
        "请复制 config.example.yaml 为 config.yaml 并填入真实配置。"
    )


def _apply_env_overrides(config: dict, prefix: str = "APP_") -> dict:
    """用环境变量覆盖配置中的敏感字段

    环境变量命名规则: APP_<一级key>__<二级key>
    例: APP_LLM__API_KEY=sk-xxx 覆盖 config["llm"]["api_key"]
    """
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        path = env_key[len(prefix):].lower()
        keys = path.split("__")
        if len(keys) < 2:
            continue
        target = config
        for k in keys[:-1]:
            if k in target:
                target = target[k]
            else:
                target = None
                break
        if target is not None and keys[-1] in target:
            target[keys[-1]] = env_val
    return config


def load_config(config_path: str | Path | None = None) -> dict:
    """加载配置"""
    global _config_cache

    if config_path:
        path = Path(config_path)
    elif _config_cache is not None:
        return _config_cache
    else:
        path = _find_config()

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config = _apply_env_overrides(config)
    _config_cache = config
    return config


def reload_config(config_path: str | Path | None = None) -> dict:
    """强制重新加载配置"""
    global _config_cache
    _config_cache = None
    return load_config(config_path)


def get_llm_config() -> dict:
    return load_config().get("llm", {})


def get_aliyun_config() -> dict:
    return load_config().get("aliyun", {})


def get_embedding_config() -> dict:
    return load_config().get("embedding", {})


def get_data_config() -> dict:
    return load_config().get("data", {})


def get_server_config() -> dict:
    return load_config().get("server", {})


def get_logging_config() -> dict:
    return load_config().get("logging", {})


def get_documents_dir() -> Path:
    data = get_data_config()
    path = Path(data.get("documents_dir", "data_dir"))
    if not path.is_absolute():
        path = Path(__file__).parent / path
    return path


def get_db_path() -> str:
    data = get_data_config()
    path = Path(data.get("database", "data_dir/conversations.db"))
    if not path.is_absolute():
        path = Path(__file__).parent / path
    return str(path)


def get_modelscope_cache() -> str:
    cfg = load_config().get("modelscope", {})
    path = Path(cfg.get("cache_dir", "./llms"))
    if not path.is_absolute():
        path = Path(__file__).parent / path
    return str(path)
