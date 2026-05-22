# What this tests?
## Tests /spend endpoints.

import pytest, time, uuid, json
import asyncio
import aiohttp


async def generate_key(session, models=[], team_id=None):
    url = "http://0.0.0.0:4000/key/generate"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {
        "models": models,
        "duration": None,
    }
    if team_id is not None:
        data["team_id"] = team_id

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


async def generate_org(session: aiohttp.ClientSession) -> dict:
    """
    Generate a new organization using the API.

    Args:
        session: aiohttp client session

    Returns:
        dict: Response containing org_id
    """
    url = "http://0.0.0.0:4000/organization/new"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}

    request_body = {
        "organization_alias": f"test-org-{uuid.uuid4()}",
    }

    async with session.post(url, headers=headers, json=request_body) as response:
        return await response.json()


async def generate_team(session: aiohttp.ClientSession, org_id: str) -> dict:
    """
    Generate a new team within an organization using the API.

    Args:
        session: aiohttp client session
        org_id: Organization ID to create the team in

    Returns:
        dict: Response containing team_id
    """
    url = "http://0.0.0.0:4000/team/new"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {"organization_id": org_id}

    async with session.post(url, headers=headers, json=data) as response:
        return await response.json()


@pytest.mark.asyncio
async def test_spend_logs_with_org_id():
    """
    - Create Organization
    - Create Team in organization
    - Create Key in organization
    - Make call (makes sure it's in spend logs)
    - Get request id from logs
    - Assert spend logs have correct org_id and team_id
    """
    async with aiohttp.ClientSession() as session:
        org_gen = await generate_org(session=session)
        print("org_gen: ", json.dumps(org_gen, indent=4, default=str))
        org_id = org_gen["organization_id"]
        team_gen = await generate_team(session=session, org_id=org_id)
        print("team_gen: ", json.dumps(team_gen, indent=4, default=str))
        team_id = team_gen["team_id"]
        key_gen = await generate_key(session=session, team_id=team_id)
        print("key_gen: ", json.dumps(key_gen, indent=4, default=str))
        key = key_gen["key"]
        response = await chat_completion(session=session, key=key)
        await asyncio.sleep(20)
        spend_logs_response = await get_spend_logs(
            session=session, request_id=response["id"]
        )
        print(
            "spend_logs_response: ",
            json.dumps(spend_logs_response, indent=4, default=str),
        )
        spend_logs_response = spend_logs_response[0]
        assert spend_logs_response["metadata"]["user_api_key_org_id"] == org_id
        assert spend_logs_response["metadata"]["user_api_key_team_id"] == team_id
        assert spend_logs_response["team_id"] == team_id


async def get_predict_spend_logs(session):
    url = "http://0.0.0.0:4000/global/predict/spend/logs"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {
        "data": [
            {
                "date": "2024-03-09",
                "spend": 200000,
                "api_key": "sk-test-mock-api-key-456",
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


@pytest.mark.skip(reason="datetime in ci/cd gets set weirdly")
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
