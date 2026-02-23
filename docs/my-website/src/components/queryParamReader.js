import React, { useState, useEffect } from 'react';

const CodeBlock = ({ token }) => {
  const codeWithToken = `
import os
from litellm import completion

# set ENV variables 
os.environ["LITELLM_TOKEN"] = '${token}'

messages = [{ "content": "Hello, how are you?","role": "user"}]

# openai call
response = completion(model="gpt-3.5-turbo", messages=messages)

# cohere call
response = completion("command-nightly", messages)
`;

  const codeWithoutToken = `
from litellm import completion

## set ENV variables
os.environ["OPENAI_API_KEY"] = "openai key"
os.environ["COHERE_API_KEY"] = "cohere key"


messages = [{ "content": "Hello, how are you?","role": "user"}]

# openai call
response = completion(model="gpt-3.5-turbo", messages=messages)

# cohere call
response = completion("command-nightly", messages)
`;
  return (
    <pre>
        {console.log("token: ", token)}
      {token ? codeWithToken : codeWithoutToken}
    </pre>
  )
}

const QueryParamReader = () => {
  const [token, setToken] = useState(null);

  useEffect(() => {
    const urlParams = new URLSearchParams(window.location.search);
    console.log("urlParams: ", urlParams)
    const token = urlParams.get('token');
    setToken(token);
  }, []);

  return (
    <div>
      <CodeBlock token={token} />
    </div>
  );
}

export default QueryParamReader;