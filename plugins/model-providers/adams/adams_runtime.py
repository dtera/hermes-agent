"""Adams provider 运行时解析器。

负责在运行时解析 Adams provider 的凭证和 URL，
将 url_template + model_id 解析为实际的 base_url，
并将 extra_headers 中的 ${ENV_VAR} 替换为实际值。

此模块作为独立的解析器，在 runtime_provider.py 的解析链中被调用，
做到高内聚低耦合，不污染源代码。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .utils import load_hermes_env, resolve_env_placeholders, resolve_url_template

logger = logging.getLogger(__name__)


def is_adams_provider(provider_name: str) -> bool:
    """判断是否为 Adams provider。"""
    return (provider_name or "").strip().lower() in {"adams", "adams-llm", "adams-proxy"}


def resolve_adams_runtime(
        *,
        model_name: str = "",
) -> Optional[Dict[str, Any]]:
    """解析 Adams provider 的运行时配置。

    Args:
        model_name: 当前使用的模型名称（用于从 models 配置中查找 model_id 等变量）

    Returns:
        包含 provider/api_mode/base_url/api_key/default_headers 的运行时字典，
        如果解析失败则返回 None。
    """
    try:
        from hermes_cli.config import load_config
        config = load_config()
    except Exception as exc:
        logger.debug("Adams: 加载配置失败: %s", exc)
        return None

    providers = config.get("providers", {})
    if not isinstance(providers, dict):
        return None

    adams_cfg = providers.get("adams")
    if not isinstance(adams_cfg, dict):
        return None

    url_template = str(adams_cfg.get("url_template", "") or "").strip()
    if not url_template:
        return None

    # 确定当前模型名称
    if not model_name:
        model_cfg = config.get("model", {})
        if isinstance(model_cfg, dict):
            model_name = str(model_cfg.get("default", "") or "").strip()

    # 从 models 配置中获取模型的变量（如 model_id）
    models = adams_cfg.get("models", {})
    if not isinstance(models, dict):
        models = {}

    # 如果配置了 model_backends_url，从远程获取模型列表并合并
    model_backends_url = adams_cfg.get("model_backends_url", "")
    if model_backends_url:
        from .utils import fetch_remote_models
        # 解析 extra_headers 中的环境变量（提前解析供请求使用）
        _env_vars = load_hermes_env()
        _extra_headers = adams_cfg.get("extra_headers", {})
        _resolved_headers: Dict[str, str] = {}
        if isinstance(_extra_headers, dict):
            for _k, _v in _extra_headers.items():
                _resolved_headers[str(_k)] = resolve_env_placeholders(str(_v), _env_vars)

        model_name_filters = adams_cfg.get("model_name_filters")
        remote_models = fetch_remote_models(
            model_backends_url=model_backends_url,
            headers=_resolved_headers,
            model_name_filters=model_name_filters,
        )
        # 远程模型合并到本地（本地配置优先）
        for _name, _vars in remote_models.items():
            if _name not in models:
                models[_name] = _vars

    model_vars: Dict[str, Any] = {}
    if isinstance(models, dict) and model_name:
        model_config = models.get(model_name)
        if isinstance(model_config, dict):
            model_vars = dict(model_config)

    if not model_vars:
        # 如果找不到模型配置，尝试使用第一个可用的模型
        if isinstance(models, dict) and models:
            first_model = next(iter(models.values()))
            if isinstance(first_model, dict):
                model_vars = dict(first_model)
                logger.debug("Adams: 模型 '%s' 未找到配置，使用第一个可用模型", model_name)

    if not model_vars:
        logger.warning("Adams: 无法找到任何模型配置")
        return None

    # 解析 url_template
    base_url = resolve_url_template(url_template, model_vars)
    if not base_url or "{" in base_url:
        logger.warning("Adams: url_template 解析不完整: %s", base_url)
        return None

    # 加载环境变量（供 api_key 和 extra_headers 共同使用）
    env_vars = load_hermes_env()

    # 解析 API key（支持 ${ENV_VAR} 占位符）
    api_key = str(adams_cfg.get("api_key", "") or "").strip()
    api_key = resolve_env_placeholders(api_key, env_vars)

    # 解析 extra_headers 中的环境变量
    extra_headers = adams_cfg.get("extra_headers", {})
    resolved_headers: Dict[str, str] = {}
    if isinstance(extra_headers, dict):
        for key, value in extra_headers.items():
            resolved_headers[str(key)] = resolve_env_placeholders(str(value), env_vars)

    # 同步更新注册的 Adams profile 实例的 default_headers，
    # 使得 agent_init.py 和 run_agent.py 中的 profile.default_headers 回退逻辑
    # 能正确获取到解析后的 headers
    try:
        from providers import get_provider_profile
        adams_profile = get_provider_profile("adams")
        if adams_profile is not None:
            adams_profile.default_headers = resolved_headers
    except Exception:
        pass

    return {
        "provider": "adams",
        "api_mode": "chat_completions",
        "base_url": base_url.rstrip("/"),
        "api_key": api_key or "no-key-required",
        "default_headers": resolved_headers,
        "source": "adams-provider",
        "requested_provider": "adams",
    }


def resolve_auxiliary_client(
        *,
        model: str = "",
        OpenAI,
        _extract_url_query_params,
        _normalize_resolved_model,
        _wrap_if_needed,
        _read_main_model,
        _to_async_client=None,
        async_mode: bool = False,
        is_vision: bool = False,
):
    """为辅助任务（title generation、compression 等）构建 Adams OpenAI client。

    将构建 client 的逻辑封装在 Adams 插件内部，
    auxiliary_client.py 只需传入必要的工具函数即可，实现高内聚低耦合。

    内部已包含完整的异常处理，保证任何情况下都安全返回，
    调用方无需额外 try/except。

    Args:
        model: 模型名称
        OpenAI: openai.OpenAI 类
        _extract_url_query_params: URL 查询参数提取函数
        _normalize_resolved_model: 模型名称规范化函数
        _wrap_if_needed: client 包装函数
        _read_main_model: 读取主模型名称的函数
        _to_async_client: 异步 client 转换函数（async_mode=True 时必须提供）
        async_mode: 是否返回异步 client
        is_vision: 是否为视觉模型

    Returns:
        (client, model) 元组，解析失败时返回 (None, None)
    """
    try:
        _adams_model = model or _read_main_model() or ""
        _adams_rt = resolve_adams_runtime(model_name=_adams_model)
        if _adams_rt is None:
            logger.warning(
                "Adams: 辅助任务运行时解析失败 (check config.yaml providers.adams)")
            return None, None

        _adams_base = _adams_rt.get("base_url", "")
        _adams_key = _adams_rt.get("api_key", "") or "no-key-required"
        _adams_headers = _adams_rt.get("default_headers") or {}

        _clean_base, _dq = _extract_url_query_params(_adams_base)
        _extra_kw = {"default_query": _dq} if _dq else {}
        if _adams_headers:
            _extra_kw["default_headers"] = _adams_headers

        client = OpenAI(api_key=_adams_key, base_url=_clean_base, **_extra_kw)
        final_model = _normalize_resolved_model(model or _adams_model, "adams")
        client = _wrap_if_needed(client, final_model, _adams_base, _adams_key)

        if async_mode and _to_async_client:
            return _to_async_client(client, final_model, is_vision=is_vision)
        return client, final_model
    except Exception as exc:
        logger.debug("Adams auxiliary client resolution failed: %s", exc)
        logger.warning(
            "Adams: 辅助任务 client 构建失败 (check config.yaml providers.adams)")
        return None, None
