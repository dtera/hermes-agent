"""Adams LLM Proxy provider plugin.

Adams 是一个 LLM 代理服务，支持通过 url_template 和 model_id 动态构建请求 URL，
并支持通过 extra_headers 配置自定义请求头（支持从 .hermes/.env 读取环境变量）。

配置示例 (.hermes/config.yaml):
    providers:
      adams:
        name: Adams LLM Proxy
        url_template: "http://example.com/service/{model_id}/v1"
        api_key: ${ADAMS_API_KEY}
        extra_headers:
          Adams-Platform-User: ${ADAMS_PLATFORM_USER}
          Adams-User-Token: ${ADAMS_USER_TOKEN}
          Adams-Business: ${ADAMS_BUSINESS}
        model_backends_url: http://example.com/api/v1/chat/backends
        model_name_filters:
        - deepseek-v4
        - glm5.1
        models:
          deepseek-v4-flash:
            model_id: 21471
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from providers import register_provider
from providers.base import ProviderProfile
from .utils import load_hermes_env, resolve_env_placeholders, resolve_url_template

logger = logging.getLogger(__name__)


class AdamsProviderProfile(ProviderProfile):
    """Adams LLM Proxy provider profile.

    支持动态 URL 模板和自定义请求头。
    通过覆写 default_headers 属性，实现延迟解析 extra_headers 中的环境变量。
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._adams_config: Optional[Dict[str, Any]] = None
        self._resolved_headers: Optional[Dict[str, str]] = None
        self._headers_initialized = False

    def _load_adams_config(self) -> Dict[str, Any]:
        """延迟加载 Adams provider 配置。"""
        if self._adams_config is not None:
            return self._adams_config

        try:
            from hermes_cli.config import load_config
            config = load_config()
            providers = config.get("providers", {})
            self._adams_config = providers.get("adams", {})
        except Exception as exc:
            logger.debug("加载 Adams 配置失败: %s", exc)
            self._adams_config = {}

        return self._adams_config

    def get_resolved_headers(self) -> Dict[str, str]:
        """获取解析后的 extra_headers（环境变量已替换）。

        此方法会延迟解析 extra_headers 中的 ${VAR_NAME} 占位符，
        并将结果缓存以避免重复解析。同时更新 self.default_headers
        以便被 agent_init.py 和 run_agent.py 中的 profile.default_headers 回退逻辑使用。
        """
        if self._resolved_headers is not None:
            return self._resolved_headers

        adams_cfg = self._load_adams_config()
        extra_headers = adams_cfg.get("extra_headers", {})
        if not isinstance(extra_headers, dict):
            self._resolved_headers = {}
            return self._resolved_headers

        env_vars = load_hermes_env()
        resolved: Dict[str, str] = {}
        for key, value in extra_headers.items():
            str_value = str(value)
            resolved[str(key)] = resolve_env_placeholders(str_value, env_vars)

        self._resolved_headers = resolved
        # 同步更新 default_headers，使得 agent_init.py 中的
        # profile.default_headers 回退逻辑能正确获取到解析后的 headers
        self.default_headers = resolved
        return self._resolved_headers

    def resolve_base_url(self, model_name: str = "") -> str:
        """根据模型名称解析实际的 base_url。

        从 url_template 和 models 配置中获取 model_id 等变量，
        替换模板中的占位符生成最终 URL。
        """
        adams_cfg = self._load_adams_config()
        url_template = adams_cfg.get("url_template", "")
        if not url_template:
            return self.base_url

        models = adams_cfg.get("models", {})
        if not isinstance(models, dict):
            return self.base_url

        # 查找当前模型的配置
        model_config = models.get(model_name, {})
        if not isinstance(model_config, dict):
            model_config = {}

        # 如果没有找到模型配置，尝试使用第一个可用的模型
        if not model_config and models:
            # 获取默认模型
            try:
                from hermes_cli.config import load_config
                config = load_config()
                model_cfg = config.get("model", {})
                default_model = model_cfg.get("default", "") if isinstance(model_cfg, dict) else ""
                if default_model and default_model in models:
                    model_config = models[default_model]
            except Exception:
                pass

        if not model_config:
            logger.warning("Adams: 未找到模型 '%s' 的配置", model_name)
            return self.base_url

        # 构建模板变量（模型配置中的所有 key-value 都可作为模板变量）
        template_vars = {k: str(v) for k, v in model_config.items()}

        return resolve_url_template(url_template, template_vars)

    def get_api_key(self) -> str:
        """获取 Adams provider 的 API key（支持 ${ENV_VAR} 占位符）。"""
        adams_cfg = self._load_adams_config()
        api_key = str(adams_cfg.get("api_key", "") or "")
        env_vars = load_hermes_env()
        return resolve_env_placeholders(api_key, env_vars)

    def invalidate_cache(self) -> None:
        """清除缓存，强制下次重新加载配置。"""
        self._adams_config = None
        self._resolved_headers = None

    # ── 通用 ProviderProfile 钩子覆写（逻辑全部留在插件内）────────────
    def resolve_runtime(
        self,
        *,
        model_name: str = "",
        model_cfg: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """覆写 ProviderProfile.resolve_runtime 通用钩子。

        将 url_template + model_id 动态解析为实际 base_url，并把
        extra_headers 中的 ${ENV_VAR} 解析后写回 default_headers 及返回的
        runtime dict。核心运行时解析链 (runtime_provider.py) 与辅助任务
        client (auxiliary_client.py) 都通过此钩子接入，无需在源代码中硬编码
        Adams 专用分支。
        """
        from .adams_runtime import resolve_adams_runtime

        return resolve_adams_runtime(model_name=model_name)

    def fetch_models(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 8.0,
    ) -> Optional[list]:
        """覆写 ProviderProfile.fetch_models：从 model_backends_url 动态获取模型列表。

        复用 model_flow.get_model_list（本地 models 配置 + 远程
        model_backends_url 合并，并按 model_name_filters 过滤）。
        """
        try:
            from .model_flow import get_model_list

            models = get_model_list()
            return list(models) or None
        except Exception as exc:  # pragma: no cover - 防御性
            logger.debug("Adams fetch_models 失败: %s", exc)
            return None

    def run_model_flow(
        self,
        config: Optional[Dict[str, Any]] = None,
        current_model: str = "",
    ) -> bool:
        """覆写 ProviderProfile.run_model_flow：`hermes model` 选中 Adams 时的交互流程。

        委托给 model_flow.run_model_flow（首次配置引导、添加模型、模型选择、
        远程模型列表合并、写回 config）。这样 Adams 出现在 `hermes model`
        provider 列表中并能完成选择，全部逻辑保留在插件内，核心仅通过通用
        钩子分发。
        """
        from .model_flow import run_model_flow as _run

        _run(config=config, current_model=current_model)
        return True


# 创建并注册 Adams provider profile 实例
adams = AdamsProviderProfile(
    name="adams",
    aliases=("adams-llm", "adams-proxy"),
    display_name="Adams LLM Proxy",
    description="Adams LLM Proxy (LLM proxy service supporting dynamic URL templates and custom request headers)",
    env_vars=("ADAMS_API_KEY"),
    base_url="",
    auth_type="extra_headers",
    supports_health_check=False,
)

register_provider(adams)
