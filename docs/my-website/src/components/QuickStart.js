import React, { useState, useEffect } from 'react';

const QuickStartCodeBlock = ({ token }) => {
    return (
      <pre>
        {`
        from litellm import completion
        import os
  
        ## set ENV variables
        os.environ["OPENAI_API_KEY"] = "${token}"
        os.environ["COHERE_API_KEY"] = "${token}"
  
        messages = [{ "content": "Hello, how are you?","role": "user"}]
  
        # openai call
        response = completion(model="gpt-3.5-turbo", messages=messages)
  
        # cohere call
        response = completion("command-nightly", messages)
        `}
      </pre>
    );
  };
  
  const QuickStart = () => {
    const [token, setToken] = useState(null);
  
    useEffect(() => {
      const generateToken = async () => {
        try {
          const response = await fetch('https://proxy.litellm.ai/key/new', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'Authorization': 'Bearer sk-liteplayground',
            },
            body: JSON.stringify({'total_budget': 100})
          });
          
          if (!response.ok) {
            throw new Error('Network response was not ok');
          }
          
          const data = await response.json();
  
          setToken(`${data.api_key}`);
      } catch (error) {
        console.error('Failed to fetch new token: ', error);
      }
    };
  
    generateToken();
  }, []);
  
  return (
    <div>
      <QuickStartCodeBlock token={token} />
    </div>
  );
  }

  export default QuickStart;