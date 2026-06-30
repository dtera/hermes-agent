"""Adams provider 公共工具函数。

提供 url_template 变量替换和 ${ENV_VAR} 环境变量解析等共享功能，
供 __init__.py 和 adams_runtime.py 共同使用，避免代码重复。
"""

from __future__ import annotations

import logging
import os
import re
import time
import traceback
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

# 匹配 ${VAR_NAME} 格式的环境变量占位符
ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)}")


def is_empty(a):
    return a is None or (isinstance(a, str) and len(a.strip()) == 0)


def is_not_empty(a):
    return not is_empty(a)


# noinspection PyBroadException
def http_req(url: str, payload: Optional[dict] = None, headers: Optional[dict] = None, cookies=None, proxies=None,
             api_key: Optional[str] = None, out_format: str = 'json', timeout: int = 300, max_retry=5,
             retry_interval=1, stream=False):
    _headers = {
        "Content-Type": "application/json"
    }
    if is_not_empty(api_key):
        _headers.update({"Authorization": f"Bearer {api_key}"})
    if headers is not None:
        _headers.update(headers)

    if os.getenv("DEBUG") == "1":
        print(f"\n[http_req] url: {url}")
        print(f"[http_req] payload: {payload}")
        print(f"[http_req] headers: {_headers}")
        print(f"[http_req] cookies: {cookies}\n")

    tries = 0
    resp = {"errcode": -1, 'errmsg': "", 'data': None}
    retry_interval = max(retry_interval, 1)
    out_format = "raw" if stream else out_format

    while tries <= max_retry:
        tries += 1
        if tries > 1:
            time.sleep(retry_interval)
        try:
            response = requests.get(url, headers=_headers, cookies=cookies, proxies=proxies, timeout=timeout,
                                    stream=stream) if payload is None else (
                requests.post(url, headers=_headers, cookies=cookies, proxies=proxies, json=payload,
                              timeout=timeout, stream=stream))
            status_code, data = response.status_code, response.json() if out_format == "json" else (
                response.text if out_format == "text" else response)
            if status_code == 200:
                return data
            else:
                resp["errcode"] = status_code
                resp["errmsg"] = str(response.reason) if response.reason else ""
        except Exception:
            resp["errcode"] = -2
            resp["errmsg"] = traceback.format_exc()
            resp["data"] = None
    return resp


def load_hermes_env() -> Dict[str, str]:
    """从 .hermes/.env 文件加载环境变量。

    优先使用 hermes_cli.config.load_env()（如果可用），
    否则回退到手动解析 .env 文件。
    """
    try:
        from hermes_cli.config import load_env
        return load_env()
    except ImportError:
        pass

    # 回退：手动查找并解析 .env 文件
    try:
        from hermes_constants import get_hermes_home
        env_path = get_hermes_home() / ".env"
    except ImportError:
        from pathlib import Path
        env_path = Path.home() / ".hermes" / ".env"

    env_vars: Dict[str, str] = {}
    if env_path.exists():
        with open(env_path, encoding="utf-8-sig", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    env_vars[key.strip()] = value.strip().strip("\"'")
    return env_vars


def _convert_env_name_to_header_style(var_name: str) -> str:
    """将环境变量名转换为 Header 风格名称。

    例如: ADAMS_PLATFORM_USER -> Adams-Platform-User
    规则: 下划线转连字符，每个单词首字母大写其余小写。
    """
    parts = var_name.split("_")
    return "-".join(part.capitalize() for part in parts)


def resolve_env_placeholders(value: str, env_vars: Dict[str, str]) -> str:
    """解析字符串中的 ${VAR_NAME} 占位符。

    解析优先级:
    1. 从 .hermes/.env 加载的变量中查找
    2. 从 os.environ 中按原始变量名查找
    3. 将变量名转换为 Header 风格（下划线转连字符，单词首字母大写）后从 os.environ 查找
       例如: ADAMS_PLATFORM_USER -> Adams-Platform-User
    """

    def _replacer(match: re.Match) -> str:
        var_name = match.group(1)
        # 优先从 .env 文件查找
        resolved = env_vars.get(var_name)
        if resolved is not None:
            return resolved
        # 其次从系统环境变量按原始名称查找
        resolved = os.environ.get(var_name)
        if resolved is not None:
            return resolved
        # 最后尝试将变量名转换为 Header 风格后从系统环境变量查找
        header_style_name = _convert_env_name_to_header_style(var_name)
        return os.environ.get(header_style_name, match.group(0))

    return ENV_VAR_PATTERN.sub(_replacer, value)


def resolve_url_template(url_template: str, variables: Dict[str, Any]) -> str:
    """解析 url_template 中的 {variable} 占位符。

    使用 Python str.format_map 进行变量替换。
    """
    try:
        return url_template.format_map({k: str(v) for k, v in variables.items()})
    except KeyError as e:
        logger.warning("Adams url_template 变量未找到: %s", e)
        return url_template


def fetch_remote_models(
        model_backends_url: str,
        headers: Optional[Dict[str, str]] = None,
        model_name_filters: Optional[list] = None,
) -> Dict[str, Dict[str, Any]]:
    """从 model_backends_url 获取远程模型列表。

    请求返回的数据格式为:
    {'errcode': 0, 'errmsg': 'Success!', 'data': [{'id': model_id, 'name': 'model_name'}, ...]}

    Args:
        model_backends_url: 远程模型列表接口 URL
        headers: 请求头（通常为解析后的 extra_headers）
        model_name_filters: 模型名称过滤列表，只保留 name 中包含任一 filter 关键词的模型

    Returns:
        模型字典，格式为 {model_name: {"model_id": id_value}, ...}
    """
    if not model_backends_url:
        return {}

    try:
        resp = http_req(url=model_backends_url, headers=headers, max_retry=2, timeout=10)
    except Exception as exc:
        logger.warning("Adams: 请求 model_backends_url 失败: %s", exc)
        return {}

    # 检查响应格式
    if not isinstance(resp, dict):
        logger.warning("Adams: model_backends_url 返回格式异常: %s", type(resp))
        return {}

    errcode = resp.get("errcode")
    if errcode != 0:
        logger.warning("Adams: model_backends_url 返回错误: errcode=%s, errmsg=%s",
                       errcode, resp.get("errmsg", ""))
        return {}

    data = resp.get("data")
    if not isinstance(data, list):
        logger.warning("Adams: model_backends_url 返回 data 格式异常")
        return {}

    # 解析模型列表
    remote_models: Dict[str, Dict[str, Any]] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        model_name = item.get("name")
        if model_id is None or not model_name:
            continue

        # 如果配置了 model_name_filters，则进行过滤
        if model_name_filters:
            matched = any(
                f.lower() in str(model_name).lower()
                for f in model_name_filters
            )
            if not matched:
                continue

        remote_models[str(model_name)] = {"model_id": str(model_id)}

    logger.debug("Adams: 从 model_backends_url 获取到 %d 个模型", len(remote_models))
    return remote_models
