import os
import logging
import urllib.parse
import urllib.request
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta
from typing import List

import json

logger = logging.getLogger()

NOCOBASE_API_URL = os.environ.get('NOCOBASE_API_URL')
NOCOBASE_API_TOKEN = os.environ.get('NOCOBASE_API_TOKEN')


def nocobase_post_json_data(path, data, max_retries: int = 3):
    headers = {
        'content-type': 'application/json',
        'authorization': f'Bearer {NOCOBASE_API_TOKEN}',
    }
    retry_count = 0
    while retry_count < max_retries:
        try:
            req = urllib.request.Request(f'{NOCOBASE_API_URL}{path}', headers=headers, data=json.dumps(data).encode('utf-8'), method='POST')
            with urllib.request.urlopen(req) as response:
                res_data = response.read().decode('utf-8')
                if response.status == 200:
                    break
                else:
                    logger.warning(f"Request failed: {res_data}")
        except Exception as e:
            logger.warning(f"Request attempt {retry_count + 1} failed: {e}")
            retry_count += 1
            if retry_count == max_retries:
                logger.error("All retry attempts failed")
                raise


def data_to_nocobase(datas, table, filter):
    for data in datas:
        nocobase_post_json_data(f'/{table}:create', data=data)

    # 核对数据量
    headers = {
        'authorization': f'Bearer {NOCOBASE_API_TOKEN}',
    }
    params = {
        "pageSize": 1,
        "page": 1,
        "filter": filter
    }
    req = urllib.request.Request(f'{NOCOBASE_API_URL}/{table}:list?{urllib.parse.urlencode(params)}', headers=headers, method='GET')
    with urllib.request.urlopen(req) as response:
        res_data = json.loads(response.read().decode('utf-8'))
        count = res_data['meta']['count']
        if count != len(datas):
            print(f'[{table}] import data size not same: {count} != {len(datas)}')


async def litellm_get_user_daily_activity_aggregated(start_date, end_date):
    from litellm.proxy._types import (
        LitellmUserRoles,
        UserAPIKeyAuth,
    )
    from litellm.proxy.management_endpoints import (
        internal_user_endpoints,
    )

    user_api_key_dict = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
    return await internal_user_endpoints.get_user_daily_activity_aggregated(start_date=start_date, end_date=end_date, user_api_key_dict=user_api_key_dict, model=None, api_key=None)

async def litellm_user_daily_activity(start_date, end_date):
    from litellm.proxy._types import (
        LitellmUserRoles,
        UserAPIKeyAuth,
    )
    from litellm.proxy.management_endpoints import (
        internal_user_endpoints,
        common_daily_activity,
    )

    def process_results(results: List[common_daily_activity.DailySpendData]):
        data = {}
        for a in results:
            if not a.breakdown or not a.breakdown.api_keys:
                continue
            for b in a.breakdown.api_keys.values():
                if b.metadata and b.metadata.key_alias and '@' in b.metadata.key_alias:
                    key_entry = {
                        'date': a.date.strftime('%Y-%m-%d'),
                        'email': b.metadata.key_alias,
                        'cost': round(b.metrics.spend, 8),
                        'api_key_reqs': b.metrics.api_requests
                    }
                    key = f"{key_entry['date']}-{key_entry['email']}"
                    if key in data:
                        data[key] = {
                            'date': data[key]['date'],
                            'email': data[key]['email'],
                            'cost': data[key]['cost'] + key_entry['cost'],  # Cumulative spend
                            'api_key_reqs': data[key]['api_key_reqs'] + key_entry['api_key_reqs']   # Cumulative api_requests
                        }
                    else:
                        data[key] = key_entry
        return list(data.values())

    user_api_key_dict = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
    res = await internal_user_endpoints.get_user_daily_activity_aggregated(start_date=start_date, end_date=end_date, user_api_key_dict=user_api_key_dict, model=None, api_key=None)
    return process_results(res.results)


async def litellm_get_users(start_date, end_date):
    from litellm.proxy.management_endpoints import internal_user_endpoints

    start_time = datetime.fromisoformat(f'{start_date}T00:00:00+08:00')
    end_time = datetime.fromisoformat(f'{end_date}T23:59:59+08:00')

    page = 0
    total_pages = 1
    page_size = 100
    results = []
    while (page < total_pages):
        page = page + 1
        res = await internal_user_endpoints.get_users(sort_by='created_at', sort_order='desc', page=page, page_size=page_size, role=None, user_ids=None, sso_user_ids=None, user_email=None, team=None)
        if total_pages == 1:
            total_pages = res["total_pages"]
        if start_time > res["users"][-1].created_at:
            total_pages = page
        data = [
            {
                "user_id": a.user_id,
                "email": a.user_email,
                "user_name": a.user_alias,
                "cost": a.spend,
                "date": a.created_at.isoformat()
            }
            for a in res["users"]
            if a.user_id != 'default_user_id' 
            and start_time <= a.created_at <= end_time
        ]
        results = results + data
    return results


async def ai_usage_to_nocobase(start_date: str | None = None, end_date: str | None = None):
    from litellm._logging import verbose_proxy_logger
    yesterday_str = (datetime.now(ZoneInfo("Asia/Shanghai")) - timedelta(days=1)).strftime('%Y-%m-%d')
    if start_date is None:
        start_date = yesterday_str
    if end_date is None:
        end_date = yesterday_str

    # 用户列表
    try:
        verbose_proxy_logger.info(f"[{datetime.now()}] LiteLLM用户采集开始，时间范围: [{start_date}, {end_date}]")
        datas = await litellm_get_users(start_date, end_date)
        verbose_proxy_logger.info(f"[{datetime.now()}] LiteLLM用户采集结束，推送nocobase开始: len={len(datas)}")
        data_to_nocobase(datas=datas, table='ai_user_litellm', filter='{"$and":[{"createdAt":{"$dateBetween":["' + start_date + '","'+ end_date +'"]}}]}')
        verbose_proxy_logger.info(f"[{datetime.now()}] LiteLLM用户推送nocobase结束")
    except Exception as e:
        verbose_proxy_logger.exception(f'LiteLLM用户采集失败: 日期[{start_date},{end_date}] {e}')

    verbose_proxy_logger.info(f"[{datetime.now()}] LiteLLM用量采集开始，时间范围: [{start_date}, {end_date}]")
    # res = await litellm_get_user_daily_activity_aggregated('2025-07-01', '2025-10-21')
    res = await litellm_get_user_daily_activity_aggregated(start_date, end_date)
    verbose_proxy_logger.info(f"[{datetime.now()}] LiteLLM用量采集结束，len={len(res.results)}")

    def metrics_data(breakdown):
        return {
            "email": breakdown.metadata.key_alias,
            "spend": round(breakdown.metrics.spend, 8),
            "api_requests": breakdown.metrics.api_requests,
            "api_failed_requests": breakdown.metrics.failed_requests,
            "api_successful_requests": breakdown.metrics.successful_requests,
            "total_tokens": breakdown.metrics.total_tokens,
            "cache_creation_input_tokens": breakdown.metrics.cache_creation_input_tokens,
            "cache_read_input_tokens": breakdown.metrics.cache_read_input_tokens,
            "completion_tokens": breakdown.metrics.completion_tokens,
            "prompt_tokens": breakdown.metrics.prompt_tokens,
        }
    
    # 用户用量
    try:
        datas = [
            {
                "date": a.date.strftime('%Y-%m-%d'),
                "email": m.metadata.key_alias,
                "cost": round(m.metrics.spend, 8),
                "api_key_reqs": m.metrics.api_requests
            }
            for a in res.results
            for m in a.breakdown.api_keys.values()
            if m.metadata.key_alias and '@' in m.metadata.key_alias
        ]
        verbose_proxy_logger.info(f"[{datetime.now()}] 推送nocobase开始: ai_usage_litellm 用户用量 len={len(datas)}")
        data_to_nocobase(datas=datas, table='ai_usage_litellm', filter='{"$and":[{"date":{"$dateBetween":["' + start_date + '","'+ end_date +'"]}}]}')
        verbose_proxy_logger.info(f"[{datetime.now()}] 推送nocobase结束: ai_usage_litellm 用户用量")
    except Exception as e:
        verbose_proxy_logger.exception(f'推送nocobase失败: ai_usage_litellm 用户用量 日期[{start_date}, {end_date}] {e}')

    # 模型用量
    try:
        datas = [
            {
                "date": a.date.strftime('%Y-%m-%d'),
                "catalog": "model",
                "type": t,
            } | metrics_data(m)
            for a in res.results
            for t, b in a.breakdown.models.items()
            for m in b.api_key_breakdown.values()
            if m.metadata.key_alias and '@' in m.metadata.key_alias
        ]
        verbose_proxy_logger.info(f"[{datetime.now()}] 推送nocobase开始: ai_usage_litellm_detail 模型 len={len(datas)}")
        data_to_nocobase(datas=datas, table='ai_usage_litellm_detail', filter='{"$and":[{"catalog":{"$eq":"model"}},{"date":{"$dateBetween":["' + start_date + '","'+ end_date +'"]}}]}')
        verbose_proxy_logger.info(f"[{datetime.now()}] 推送nocobase结束: ai_usage_litellm_detail 模型")
    except Exception as e:
        verbose_proxy_logger.exception(f'推送nocobase失败: ai_usage_litellm_detail 模型 日期[{start_date}, {end_date}] {e}')

    # MCP用量
    try:
        datas = [
            {
                "date": a.date.strftime('%Y-%m-%d'),
                "catalog": "mcp_server",
                "type": t,
            } | metrics_data(m)
            for a in res.results
            for t, b in a.breakdown.mcp_servers.items()
            for m in b.api_key_breakdown.values()
            if m.metadata.key_alias and '@' in m.metadata.key_alias
        ]
        verbose_proxy_logger.info(f"[{datetime.now()}] 推送nocobase开始: ai_usage_litellm_detail mcp len={len(datas)}")
        data_to_nocobase(datas=datas, table='ai_usage_litellm_detail', filter='{"$and":[{"catalog":{"$eq":"mcp_server"}},{"date":{"$dateBetween":["' + start_date + '","'+ end_date +'"]}}]}')
        verbose_proxy_logger.info(f"[{datetime.now()}] 推送nocobase结束: ai_usage_litellm_detail mcp")
    except Exception as e:
        verbose_proxy_logger.exception(f'推送nocobase失败: ai_usage_litellm_detail mcp 日期[{start_date}, {end_date}] {e}')


if __name__ == "__main__":
    datas = [
        {
            "user_id": "1122334455",
            "email": "demo@fzzixun.com",
            "user_name": "demo",
            "cost": "2.0",
            "date": "2025-10-10T05:34:38.880Z"
        }
    ]
    data_to_nocobase(datas=datas, table='ai_usage_litellm', filter='{"$and":[{"date":{"$dateOn":"2025-10-10"}}]}')
