import openai
import asyncio


async def async_request(client, model, input_data):
    response = await client.embeddings.create(model=model, input=input_data)
    response = response.dict()
    data_list = response["data"]
    for i, embedding in enumerate(data_list):
        embedding["embedding"] = []
        current_index = embedding["index"]
        assert i == current_index
    return response


async def main():
    client = openai.AsyncOpenAI(api_key="sk-1234", base_url="http://0.0.0.0:4000")
    models = [
        "text-embedding-ada-002",
        "text-embedding-ada-002",
        "text-embedding-ada-002",
    ]
    inputs = [
        [
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
            "18",
            "19",
            "20",
        ],
        ["1", "2", "3", "4", "5", "6"],
        [
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
            "18",
            "19",
            "20",
        ],
        [
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
            "18",
            "19",
            "20",
        ],
        [
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
            "18",
            "19",
            "20",
        ],
        ["1", "2", "3"],
    ]

    tasks = []
    for model, input_data in zip(models, inputs):
        task = async_request(client, model, input_data)
        tasks.append(task)

    responses = await asyncio.gather(*tasks)
    print(responses)
    for response in responses:
        data_list = response["data"]
        for embedding in data_list:
            embedding["embedding"] = []
        print(response)


asyncio.run(main())
