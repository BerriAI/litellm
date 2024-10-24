# What this tests?
## Tests /spend endpoints.

import pytest, time, uuid
import asyncio
import aiohttp


async def generate_key(session, models=[]):
    url = "http://0.0.0.0:4000/key/generate"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {
        "models": models,
        "duration": None,
    }

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")
        return await response.json()


async def chat_completion(session, key, model="gpt-3.5-turbo"):
    url = "http://0.0.0.0:4000/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Hello! {uuid.uuid4()}"},
        ],
    }

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")

        return await response.json()


async def chat_completion_high_traffic(session, key, model="gpt-3.5-turbo"):
    url = "http://0.0.0.0:4000/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Hello! {uuid.uuid4()}"},
        ],
    }
    try:
        async with session.post(url, headers=headers, json=data) as response:
            status = response.status
            response_text = await response.text()

            if status != 200:
                raise Exception(f"Request did not return a 200 status code: {status}")

            return await response.json()
    except Exception as e:
        return None


async def get_spend_logs(session, request_id=None, api_key=None):
    if api_key is not None:
        url = f"http://0.0.0.0:4000/spend/logs?api_key={api_key}"
    else:
        url = f"http://0.0.0.0:4000/spend/logs?request_id={request_id}"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}

    async with session.get(url, headers=headers) as response:
        status = response.status
        response_text = await response.text()

        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")
        return await response.json()


@pytest.mark.asyncio
async def test_spend_logs():
    """
    - Create key
    - Make call (makes sure it's in spend logs)
    - Get request id from logs
    """
    async with aiohttp.ClientSession() as session:
        key_gen = await generate_key(session=session)
        key = key_gen["key"]
        response = await chat_completion(session=session, key=key)
        await asyncio.sleep(20)
        await get_spend_logs(session=session, request_id=response["id"])


async def get_predict_spend_logs(session):
    url = f"http://0.0.0.0:4000/global/predict/spend/logs"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {
        "data": [
            {
                "date": "2024-03-09",
                "spend": 200000,
                "api_key": "f19bdeb945164278fc11c1020d8dfd70465bffd931ed3cb2e1efa6326225b8b7",
            }
        ]
    }

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")
        return await response.json()


async def get_spend_report(session, start_date, end_date):
    url = "http://0.0.0.0:4000/global/spend/report"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    async with session.get(
        url, headers=headers, params={"start_date": start_date, "end_date": end_date}
    ) as response:
        status = response.status
        response_text = await response.text()

        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")
        return await response.json()


@pytest.mark.asyncio
async def test_get_predicted_spend_logs():
    """
    - Create key
    - Make call (makes sure it's in spend logs)
    - Get request id from logs
    """
    async with aiohttp.ClientSession() as session:
        result = await get_predict_spend_logs(session=session)
        print(result)

        assert "response" in result
        assert len(result["response"]) > 0


@pytest.mark.skip(reason="High traffic load test, meant to be run locally")
@pytest.mark.asyncio
async def test_spend_logs_high_traffic():
    """
    - Create key
    - Make 30 concurrent calls
    - Get all logs for that key
    - Wait 10s
    - Assert it's 30
    """

    async def retry_request(func, *args, _max_attempts=5, **kwargs):
        for attempt in range(_max_attempts):
            try:
                return await func(*args, **kwargs)
            except (
                aiohttp.client_exceptions.ClientOSError,
                aiohttp.client_exceptions.ServerDisconnectedError,
            ) as e:
                if attempt + 1 == _max_attempts:
                    raise  # re-raise the last ClientOSError if all attempts failed
                print(f"Attempt {attempt+1} failed, retrying...")

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=600)
    ) as session:
        start = time.time()
        key_gen = await generate_key(session=session)
        key = key_gen["key"]
        n = 1000
        tasks = [
            retry_request(
                chat_completion_high_traffic,
                session=session,
                key=key,
                model="azure-gpt-3.5",
            )
            for _ in range(n)
        ]
        chat_completions = await asyncio.gather(*tasks)
        successful_completions = [c for c in chat_completions if c is not None]
        print(f"Num successful completions: {len(successful_completions)}")
        await asyncio.sleep(10)
        try:
            response = await retry_request(get_spend_logs, session=session, api_key=key)
            print(f"response: {response}")
            print(f"len responses: {len(response)}")
            assert len(response) == n
            print(n, time.time() - start, len(response))
        except Exception:
            print(n, time.time() - start, 0)
        raise Exception("it worked!")


@pytest.mark.asyncio
async def test_spend_report_endpoint():
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=600)
    ) as session:
        import datetime

        todays_date = datetime.date.today() + datetime.timedelta(days=1)
        todays_date = todays_date.strftime("%Y-%m-%d")

        print("todays_date", todays_date)
        thirty_days_ago = (
            datetime.date.today() - datetime.timedelta(days=30)
        ).strftime("%Y-%m-%d")
        spend_report = await get_spend_report(
            session=session, start_date=thirty_days_ago, end_date=todays_date
        )
        print("spend report", spend_report)

        for row in spend_report:
            date = row["group_by_day"]
            teams = row["teams"]
            for team in teams:
                team_name = team["team_name"]
                total_spend = team["total_spend"]
                metadata = team["metadata"]

                assert team_name is not None

                print(f"Date: {date}")
                print(f"Team: {team_name}")
                print(f"Total Spend: {total_spend}")
                print("Metadata: ", metadata)
                print()
