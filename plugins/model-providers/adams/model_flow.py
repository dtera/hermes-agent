"""Adams provider 模型选择流程。

将 hermes model 命令中 Adams 专用的模型选择逻辑封装在此模块中，
做到高内聚低耦合，避免污染 main.py 等源代码文件。

对外暴露两个接口：
- get_model_list(): 获取 Adams 配置中的模型列表（供 model_switch.py 使用）
- run_model_flow(): 执行 Adams 的模型选择交互流程（供 main.py 使用）
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 匹配 url_template 中的 {variable} 占位符
_URL_TEMPLATE_VAR_PATTERN = re.compile(r"\{([^}]+)}")


def _load_full_config() -> Dict[str, Any]:
    """加载完整配置。"""
    from hermes_cli.config import load_config
    return load_config()


def _load_adams_provider_config() -> Dict:
    """加载 Adams provider 配置。"""
    try:
        cfg = _load_full_config()
        return cfg.get("providers", {}).get("adams", {})
    except Exception as exc:
        logger.debug("Adams model_flow: 加载配置失败: %s", exc)
        return {}


def _extract_url_template_vars(url_template: str) -> List[str]:
    """从 url_template 中提取所有 {variable} 变量名。

    例如: "http://example.com/service/{model_id}/v1" -> ["model_id"]
    """
    return _URL_TEMPLATE_VAR_PATTERN.findall(url_template)


def _resolve_extra_headers(adams_cfg: Dict[str, Any]) -> Dict[str, str]:
    """解析 adams 配置中 extra_headers 的 ${ENV_VAR} 占位符。"""
    from .utils import load_hermes_env, resolve_env_placeholders

    env_vars = load_hermes_env()
    extra_headers = adams_cfg.get("extra_headers", {})
    resolved: Dict[str, str] = {}
    if isinstance(extra_headers, dict):
        for key, value in extra_headers.items():
            resolved[str(key)] = resolve_env_placeholders(str(value), env_vars)
    return resolved


def _fetch_remote_models_for(adams_cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """根据 adams 配置从 model_backends_url 拉取远程模型。

    自动解析 extra_headers 中的环境变量作为请求头，并按 model_name_filters
    过滤。未配置 model_backends_url 时返回空字典。
    """
    url = str(adams_cfg.get("model_backends_url") or "").strip()
    if not url:
        return {}
    from .utils import fetch_remote_models

    return fetch_remote_models(
        model_backends_url=url,
        headers=_resolve_extra_headers(adams_cfg),
        model_name_filters=adams_cfg.get("model_name_filters"),
    )


def _merge_models(adams_cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """合并本地 models 配置与远程 model_backends_url 模型（本地配置优先）。"""
    local = adams_cfg.get("models", {})
    if not isinstance(local, dict):
        local = {}
    merged: Dict[str, Dict[str, Any]] = dict(local)
    for name, vars_dict in _fetch_remote_models_for(adams_cfg).items():
        merged.setdefault(name, vars_dict)
    return merged


def get_model_list() -> List[str]:
    """获取 Adams provider 配置中的模型名称列表。

    从 config.yaml 的 providers.adams.models 中读取，
    如果配置了 model_backends_url，还会从远程获取模型列表并合并。
    供 model_switch.py 的 list_authenticated_providers 使用。

    Returns:
        模型名称列表，如 ['deepseek-v4-flash', 'deepseek-v4-pro', 'glm-5.1']
    """
    return list(_merge_models(_load_adams_provider_config()).keys())


def _setup_adams_provider() -> Optional[Dict[str, Any]]:
    """首次配置 Adams provider 的引导流程。

    引导用户输入：
    1. url_template（如 http://example.com/service/{model_id}/v1）
    2. api_key（写入 .env 的 ADAMS_API_KEY 中，yaml 配置 ${ADAMS_API_KEY} 引用）
    3. extra_headers（JSON 字符串，值写入 .env 中，key 为 JSON 中对应的 key，yaml 配置 ${key} 引用）

    Returns:
        配置好的 adams provider 字典，或 None（用户取消）
    """
    from hermes_cli.config import save_env_value

    print("  ╭─ Adams LLM Proxy 初始配置 ─────────────────────────────────────────╮")
    print("  │ 首次使用 Adams provider，需要进行以下配置：                             ")
    print("  ╰────────────────────────────────────────────────────────────────────╯")
    print()

    # 1. 输入 url_template
    print("  📌 请输入 URL 模板（支持 {变量} 占位符）")
    print("     示例: http://example.com/service/{model_id}/v1")
    url_template = input("  url_template: ").strip()
    if not url_template:
        print("  ❌ 已取消配置。")
        return None
    print()

    # 2. 输入 api_key（写入 .env）
    print("  🔑 请输入 API Key（将安全存储到 .hermes/.env 中）")
    api_key = input("  api_key: ").strip()
    if not api_key:
        print("  ❌ 已取消配置。")
        return None

    # 保存到 .env
    save_env_value("ADAMS_API_KEY", api_key)
    print("     ✅ 已保存到 .hermes/.env (ADAMS_API_KEY)")
    print()

    # 3. 输入 extra_headers（JSON 字符串）
    print("  📋 请输入额外请求头（JSON 格式，值将安全存储到 .hermes/.env 中）")
    print(
        "     示例: {\"Adams-Platform-User\": \"user\", \"Adams-User-Token\": \"your-token\", \"Adams-Business\": \"3873\"}")
    print("     直接回车跳过（无额外请求头）")
    headers_input = input("  extra_headers (JSON): ").strip()

    extra_headers_yaml: Dict[str, str] = {}
    if headers_input:
        try:
            headers_dict = json.loads(headers_input)
            if not isinstance(headers_dict, dict):
                print("  ⚠️  JSON 格式错误，应为对象格式 {}，已跳过。")
            else:
                for key, value in headers_dict.items():
                    str_value = str(value).strip()
                    # 将 header 的 key 转换为环境变量名（大写 + 下划线）
                    env_key = re.sub(r"[^A-Za-z0-9]", "_", key).upper()
                    # 保存值到 .env
                    save_env_value(env_key, str_value)
                    # yaml 中使用 ${ENV_KEY} 引用
                    extra_headers_yaml[key] = f"${{{env_key}}}"
                print(f"     ✅ 已保存 {len(extra_headers_yaml)} 个 header 值到 .hermes/.env")
        except json.JSONDecodeError as e:
            print(f"  ⚠️  JSON 解析失败: {e}，已跳过。")
    print()

    # 4. 可选：model_backends_url（动态模型列表接口）
    print("  🌐 (可选) 模型列表接口 model_backends_url")
    print("     配置后将从该接口动态拉取可用模型，无需手动维护 model_id。")
    print("     示例: http://example.com/api/v1/chat/backends")
    print("     直接回车跳过")
    backends_url = input("  model_backends_url: ").strip()
    print()

    # 5. 可选：model_name_filters（仅在配置了 backends_url 时询问）
    model_name_filters: List[str] = []
    if backends_url:
        print("  🔎 (可选) 模型名称过滤（英文逗号分隔关键词，仅保留名称含任一关键词的模型）")
        print("     示例: deepseek-v4, glm5.1, hy3-preview")
        print("     直接回车则不过滤")
        filters_input = input("  model_name_filters: ").strip()
        if filters_input:
            model_name_filters = [f.strip() for f in filters_input.split(",") if f.strip()]
        print()

    # 构建 Adams provider 配置
    adams_cfg: Dict[str, Any] = {
        "name": "Adams LLM Proxy",
        "url_template": url_template,
        "api_key": "${ADAMS_API_KEY}",
    }
    if extra_headers_yaml:
        adams_cfg["extra_headers"] = extra_headers_yaml
    if backends_url:
        adams_cfg["model_backends_url"] = backends_url
        if model_name_filters:
            adams_cfg["model_name_filters"] = model_name_filters

    return adams_cfg


def _add_model_to_config(adams_cfg: Dict[str, Any]) -> Optional[str]:
    """添加模型到 Adams provider 配置。

    根据 url_template 中的变量，引导用户输入 model_name 和对应变量值。

    Args:
        adams_cfg: 当前 Adams provider 配置

    Returns:
        新添加的模型名称，或 None（用户取消）
    """
    url_template = adams_cfg.get("url_template", "")
    template_vars = _extract_url_template_vars(url_template)

    print("  ╭─ 添加模型 ────────────────────────────────────────────────────────╮")
    if url_template:
        print(f"  │ URL 模板: {url_template}")
        if template_vars:
            print(f"  │ 模板变量: {', '.join(template_vars)}")
    print("  ╰───────────────────────────────────────────────────────────────────╯")
    print()

    # 输入模型名称
    print("  📌 请输入模型名称（如 deepseek-v4-flash）")
    model_name = input("  model_name: ").strip()
    if not model_name:
        print("  ❌ 已取消。")
        return None

    # 输入 url_template 中的变量值
    model_vars: Dict[str, Any] = {}
    if template_vars:
        print()
        print(f"  📌 请输入模板变量值（用于替换 URL 中的占位符）")
        for var_name in template_vars:
            value = input(f"  {var_name}: ").strip()
            if not value:
                print(f"  ❌ 变量 {var_name} 不能为空，已取消。")
                return None
            # 尝试转换为数字
            try:
                model_vars[var_name] = int(value)
            except ValueError:
                model_vars[var_name] = value

    # 写入配置
    models = adams_cfg.get("models", {})
    if not isinstance(models, dict):
        models = {}
    models[model_name] = model_vars
    adams_cfg["models"] = models

    print()
    print(f"  ✅ 模型 '{model_name}' 已添加")
    if model_vars:
        for k, v in model_vars.items():
            print(f"     {k}: {v}")

    return model_name


def run_model_flow(config: Optional[Dict] = None, current_model: str = "") -> None:
    """Adams LLM Proxy 专用的 model 选择流程。

    流程：
    1. 如果 Adams provider 未配置 → 引导首次配置
    2. 合并本地 models 与远程 model_backends_url 模型
    3. 远程拉取失败时给出诊断并支持刷新/手动添加
    4. 显示模型列表让用户选择（支持刷新、手动输入）

    Args:
        config: 当前配置字典（未使用，保持接口兼容）
        current_model: 当前选中的模型名称
    """
    from hermes_cli.auth import _prompt_model_selection, deactivate_provider
    from hermes_cli.config import load_config, save_config

    cfg = load_config()

    # 确保 providers 字典存在
    if "providers" not in cfg:
        cfg["providers"] = {}
    providers = cfg["providers"]

    # ── 步骤1: 检查 Adams provider 是否已配置 ──
    adams_cfg = providers.get("adams")
    if not adams_cfg or not isinstance(adams_cfg, dict):
        adams_cfg = _setup_adams_provider()
        if adams_cfg is None:
            return
        providers["adams"] = adams_cfg
        save_config(cfg)
        # 重新加载配置（save_config 后缓存失效）
        cfg = load_config()
        adams_cfg = cfg.get("providers", {}).get("adams", {})

    name = adams_cfg.get("name", "Adams LLM Proxy")
    url_template = adams_cfg.get("url_template", "")
    backends_url = str(adams_cfg.get("model_backends_url") or "").strip()

    refresh_option = "🔄 刷新模型列表"
    add_model_option = "➕ 手动输入模型 (model_id)..."

    def _persist_adams() -> None:
        """将内存中的 adams_cfg 写回并重新加载（用于添加模型后刷新）。"""
        nonlocal cfg, adams_cfg
        cfg["providers"]["adams"] = adams_cfg
        save_config(cfg)
        cfg = load_config()
        adams_cfg = cfg.get("providers", {}).get("adams", {})

    # ── 步骤2: 选择模型（支持刷新远程列表的循环）──
    while True:
        local_models = adams_cfg.get("models", {})
        local_count = len(local_models) if isinstance(local_models, dict) else 0
        merged = _merge_models(adams_cfg)
        model_list = sorted(merged.keys())

        # 无可用模型：区分"远程拉取失败" vs "确实未配置"
        if not model_list:
            if backends_url:
                print(f"  ⚠️  未能从 model_backends_url 获取到模型：{backends_url}")
                print("     可能原因：网络不可达、凭证/请求头无效，或被 model_name_filters 过滤为空。")
                print("     可检查 .hermes/.env 中的请求头变量，或确认内网连通性后刷新。")
                print()
                choice = _prompt_model_selection(
                    [refresh_option, add_model_option], current_model=""
                )
                if choice == refresh_option:
                    print("  🔄 正在重新拉取模型列表...")
                    print()
                    continue
                if choice == add_model_option:
                    if _add_model_to_config(adams_cfg) is None:
                        return
                    _persist_adams()
                    continue
                print("  No change.")
                return
            else:
                print("  ⚠️  Adams provider 尚未配置任何模型，请先添加模型。")
                print()
                if _add_model_to_config(adams_cfg) is None:
                    return
                _persist_adams()
                continue

        # 显示当前配置信息
        print(f"  Provider: {name}")
        if url_template:
            print(f"  URL template: {url_template}")
        if backends_url:
            remote_count = max(len(model_list) - local_count, 0)
            print(
                f"  Available models: {len(model_list)} "
                f"(本地 {local_count} + 远程 {remote_count})"
            )
        else:
            print(f"  Available models: {len(model_list)}")
        print()

        # 组装选项：模型列表 + 刷新（仅远程模式）+ 手动输入
        display_list = list(model_list)
        if backends_url:
            display_list.append(refresh_option)
        display_list.append(add_model_option)

        selected = _prompt_model_selection(display_list, current_model=current_model)

        if selected == refresh_option:
            print("  🔄 正在重新拉取模型列表...")
            print()
            continue
        if selected == add_model_option:
            print()
            model_name = _add_model_to_config(adams_cfg)
            if model_name is None:
                return
            cfg["providers"]["adams"] = adams_cfg
            selected = model_name
        elif not selected:
            print("  No change.")
            return

        # ── 统一保存模型选择 ──
        # 一次性设置 model.default 和 model.provider，避免双重 save 覆盖
        model_cfg = cfg.get("model")
        if not isinstance(model_cfg, dict):
            model_cfg = {"default": model_cfg} if model_cfg else {}
            cfg["model"] = model_cfg
        model_cfg["default"] = selected
        model_cfg["provider"] = "adams"
        # Adams 使用 url_template 动态生成 base_url，不需要静态 base_url
        model_cfg.pop("base_url", None)
        model_cfg.pop("api_mode", None)
        save_config(cfg)
        deactivate_provider()

        print(f"  ✅ Default model set to: {selected} (via {name})")
        return
