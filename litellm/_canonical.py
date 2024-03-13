import requests
import json
import aiohttp
from typing import Dict, Any
from litellm._logging import verbose_logger
import litellm
from typing import Optional

DEFAULT_CACHE_HOST = "https://cacheapp.canonical.chat/"
def print_verbose(print_statement):
    try:
        verbose_logger.debug(print_statement)
        if litellm.set_verbose:
            print(print_statement)  # noqa
    except:
        pass

def url(host: str) -> str:
    return f"{host}api/v1/cache"

def headers(apikey: str) -> Dict[str, str]:
    return {
        "Content-Type": "application/json",
        "X-Canonical-Api-Key": apikey,
    }

class CanonicalClient():

    def __init__(self, host: Optional[str], api_key: str) -> None:
        if api_key is None:
            raise ValueError("An API key is required to use the Canonical Neural Cache API. To get one, contact Canonical: founders@canonical.chat.")
        self.apikey = api_key
        if host is None:
          self.host = DEFAULT_CACHE_HOST
        if not self.host.endswith("/"):
            self.host += "/"

    # GET
    def get(self, key: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        key["stream"] = False
        response = requests.post(
            url=f"{self.host}chat/completions",
            headers=headers(self.apikey),
            data=json.dumps(key),
        )
        if response.status_code == 200:
            return response.json()
        return None

    async def async_get(self, key: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        key["stream"] = False
        async with aiohttp.ClientSession() as session:
            async with session.post(url=f"{self.host}chat/completions",
                                    json=key,
                                    headers=headers(self.apikey)) as response:
                if response.status == 200:
                    return await response.json()
                return None

    # SET
    def set(self, key: Dict[str, Any], value: Dict[str, Any]) -> None:
      messages = key.get("messages", [])
      jsonresponse = json.loads(value["response"])
      messages.append(jsonresponse["choices"][0]["message"])
      key["messages"] = messages
      _ = requests.request(
          method="POST",
          url=url(self.host),
          headers=headers(self.apikey),
          data=json.dumps(key),
      )

    async def async_set(self, key: Dict[str, Any], value: Dict[str, Any]) -> None:
      messages = key.get("messages", [])
      messages.append(value["response"]["choices"][0]["message"])
      key["messages"] = messages
      async with aiohttp.ClientSession() as session:
          async with session.post(url(self.host),
                                  json=key,
                                  headers=headers(self.apikey)) as response:
                _ = await response.json()
