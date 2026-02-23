export const CREATE_END_USER_CURL_COMMAND = `
curl -X POST --location '<your_proxy_base_url>/end_user/new' \\

-H 'Authorization: Bearer <your-master-key>' \\

-H 'Content-Type: application/json' \\

-d '{"user_id": "my-customer-id', "budget_id": "<BUDGET_ID>"}' # 👈 KEY CHANGE

`;

export const CHAT_COMPLETIONS_CURL_COMMAND = `
curl -X POST --location '<your_proxy_base_url>/chat/completions' \\

-H 'Authorization: Bearer <your-master-key>' \\

-H 'Content-Type: application/json' \\

-d '{
  "model": "gpt-3.5-turbo',
  "messages":[{"role": "user", "content": "Hey, how's it going?"}],
  "user": "my-customer-id"
}' # 👈 KEY CHANGE

`;

export const OPENAI_SDK_PYTHON_CODE = `from openai import OpenAI
client = OpenAI(
  base_url="<your_proxy_base_url>",
  api_key="<your_proxy_key>"
)

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  user="my-customer-id"
)

print(completion.choices[0].message)`;
