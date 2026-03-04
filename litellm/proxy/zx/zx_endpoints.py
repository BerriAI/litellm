import os
import logging
from zoneinfo import ZoneInfo
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from litellm._logging import verbose_proxy_logger
from . import zx_user_endpoints

logger = logging.getLogger()


def scheduler_start(scheduler: AsyncIOScheduler):
    from . import zx_job

    if zx_job.NOCOBASE_API_URL and zx_job.NOCOBASE_API_TOKEN:
        # 添加定时任务 - 每天上午5点执行
        scheduler.add_job(
            zx_job.ai_usage_to_nocobase,
            trigger=CronTrigger(hour=5, minute=0, timezone=ZoneInfo("Asia/Shanghai")),
            id="ai_usage_to_nocobase",
            name="推送AI使用数据到nocobase - 每天上午5点执行",
        )
        verbose_proxy_logger.info(f"推送AI使用数据到nocobase定时任务 - 每天上午5点执行")
    else:
        verbose_proxy_logger.info(
            f"推送AI使用数据到nocobase定时任务 - 未开启，NOCOBASE_API_URL或NOCOBASE_API_TOKEN环境变量未配置"
        )
        return

    # scheduler.add_job(
    #     zx_job.ai_usage_to_nocobase,
    #     "interval",
    #     seconds=60,
    #     # "cron",
    #     # hour=17,
    #     # minute=10,
    #     # timezone=ZoneInfo("Asia/Shanghai"),
    # )


# 定时任务集合
jobs = [scheduler_start]

# 路由配置
routers = [zx_user_endpoints.router]

ZX_LLM_DEVEOPER_ENABLED = os.environ.get("ZX_LLM_DEVEOPER_ENABLED") == "true"
if ZX_LLM_DEVEOPER_ENABLED:
    check = True
    required_params = [
        "ZX_AUTH_HOST",
        "ZX_AUTH_API_HOST",
        "ZX_AUTH_APP_KEY",
        "ZX_AUTH_APP_SECRET",
    ]
    for param_name in required_params:
        if not os.environ.get(param_name):
            check = False
            logger.warning(f"缺少必要参数：{param_name}")
    if check:
        from . import zx_config_endpoints

        routers.append(zx_config_endpoints.router)
    else:
        logger.warning(f"缺少必要参数，不添加zx_config_endpoints")
