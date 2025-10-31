import os
import logging
import urllib.parse
import urllib.request
from typing import List

import json

logger = logging.getLogger()

SERVER_URL = os.environ.get('NOCOBASE_API_URL', 'https://test-etools-nocobase.fzzixun.com/nocobase/api')
SERVER_TOKEN = os.environ.get('NOCOBASE_API_TOKEN')


def nocobase_post_json_data(path, data, max_retries: int = 3):
    headers = {
        'content-type': 'application/json',
        'authorization': f'Bearer {SERVER_TOKEN}',
    }
    retry_count = 0
    while retry_count < max_retries:
        try:
            req = urllib.request.Request(f'{SERVER_URL}{path}', headers=headers, data=json.dumps(data).encode('utf-8'), method='POST')
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

async def litellm_data_user_daily_activity():
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
                        'date': a.date, 
                        'email': b.metadata.key_alias, 
                        'cost': b.metrics.spend, 
                        'api_requests': b.metrics.api_requests
                    }
                    key = f"{key_entry['date']}-{key_entry['email']}"
                    if key in data:
                        data[key] = {
                            'date': data[key]['date'],
                            'email': data[key]['email'],
                            'cost': data[key]['cost'] + key_entry['cost'],  # Cumulative spend
                            'api_requests': data[key]['api_requests'] + key_entry['api_requests']   # Cumulative api_requests
                        }
                    else:
                        data[key] = key_entry
        return list(data.values())

    user_api_key_dict = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
    start_date = ''
    end_date = ''
    page = 0
    total_pages = 1
    page_size = 1000
    results: List[common_daily_activity.DailySpendData] = []
    while (page < total_pages):
        page = page + 1
        res = await internal_user_endpoints.get_user_daily_activity(start_date=start_date, end_date=end_date, page=page, page_size=page_size, user_api_key_dict=user_api_key_dict)
        if total_pages == 1:
            total_pages = res.metadata.total_pages
        results = results + res.results
    return process_results(results)


def ai_usage_to_nocobase(datas):
    for data in datas:
        nocobase_post_json_data('/ai_usage_litellm:create', data=data)

    # 核对数据量
    headers = {
        'authorization': f'Bearer {SERVER_TOKEN}',
    }
    params = {
        "pageSize": 1,
        "page": 1,
        "filter": '{"$and":[{"date":{"$dateOn":"2025-10-10"}}]}'
    }
    req = urllib.request.Request(f'{SERVER_URL}/ai_usage_litellm:list?{urllib.parse.urlencode(params)}', headers=headers, method='GET')
    with urllib.request.urlopen(req) as response:
        res_data = json.load(response.read().decode('utf-8'))
        if res_data.meta.count != len(datas):
            logger.error(f'import data size not same: {res_data.meta.count} != {len(datas)}')


async def litellm_data_ai_usage_to_nocobase():
    datas = await litellm_data_user_daily_activity()
    ai_usage_to_nocobase(datas=datas)


if __name__ == "__main__":
    datas = [{
        "email": "string",
        "cost": "1.0",
        "date": "2025-10-10T05:34:38.880Z",
        "api_key_reqs": "100"
    }]
    ai_usage_to_nocobase(datas=datas)
