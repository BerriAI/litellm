import React, { useState, useEffect } from 'react';

const CodeBlock = ({ token }) => {
  const codeWithToken = `${token}`;

  return (
    <pre>
      {token ? codeWithToken : ""}
    </pre>
  );
};

const TokenGen = () => {
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
    <CodeBlock token={token} />
  </div>
);
};

export default TokenGen;
