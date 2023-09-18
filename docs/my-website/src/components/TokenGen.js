import React, { useState, useEffect } from 'react';
import {v4 as uuidv4} from 'uuid'; 

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
    const generateToken = () => {
      // Generate a special uuid/token "sk-litellm-<uuid>"
      const newToken = `sk-litellm-${uuidv4()}`;
      setToken(newToken);
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