import React, { useState } from 'react';
import styles from './transform_request.module.css';

const DEFAULT_REQUEST = {
  "model": "bedrock/gpt-4",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Explain quantum computing in simple terms"
    }
  ],
  "temperature": 0.7,
  "max_tokens": 500,
  "stream": true
};

type ViewMode = 'split' | 'request' | 'transformed';

const TransformRequestPlayground: React.FC = () => {
  const [request, setRequest] = useState(JSON.stringify(DEFAULT_REQUEST, null, 2));
  const [transformedRequest, setTransformedRequest] = useState('');
  const [viewMode, setViewMode] = useState<ViewMode>('split');

  const handleTransform = async () => {
    try {
      // Here you would make the actual API call to transform the request
      // For now, we'll just set a sample response
      const sampleResponse = `curl -X POST \\
  https://api.openai.com/v1/chat/completions \\
  -H 'Authorization: Bearer sk-xxx' \\
  -H 'Content-Type: application/json' \\
  -d '{
    "model": "gpt-4",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      }
    ],
    "temperature": 0.7
  }'`;
      setTransformedRequest(sampleResponse);
    } catch (error) {
      console.error('Error transforming request:', error);
    }
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(transformedRequest);
  };

  const renderContent = () => {
    switch (viewMode) {
      case 'request':
        return (
          <div className={styles.panel}>
            <div className={styles['panel-header']}>
              <h2>Original Request</h2>
              <p>The request you would send to LiteLLM /chat/completions endpoint.</p>
            </div>
            <textarea
              className={styles['code-input']}
              value={request}
              onChange={(e) => setRequest(e.target.value)}
              spellCheck={false}
            />
            <div className={styles['panel-footer']}>
              <button className={styles['transform-button']} onClick={handleTransform}>
                Transform →
              </button>
            </div>
          </div>
        );
      case 'transformed':
        return (
          <div className={styles.panel}>
            <div className={styles['panel-header']}>
              <h2>Transformed Request</h2>
              <p>How LiteLLM transforms your request for the specified provider.</p>
              <p className={styles.note}>Note: Sensitive headers are not shown.</p>
            </div>
            <div className={styles['code-output-container']}>
              <pre className={styles['code-output']}>{transformedRequest}</pre>
              <button className={styles['copy-button']} onClick={handleCopy}>
                Copy
              </button>
            </div>
          </div>
        );
      default:
        return (
          <>
            <div className={styles.panel}>
              <div className={styles['panel-header']}>
                <h2>Original Request</h2>
                <p>The request you would send to LiteLLM /chat/completions endpoint.</p>
              </div>
              <textarea
                className={styles['code-input']}
                value={request}
                onChange={(e) => setRequest(e.target.value)}
                spellCheck={false}
              />
              <div className={styles['panel-footer']}>
                <button className={styles['transform-button']} onClick={handleTransform}>
                  Transform →
                </button>
              </div>
            </div>
            <div className={styles.panel}>
              <div className={styles['panel-header']}>
                <h2>Transformed Request</h2>
                <p>How LiteLLM transforms your request for the specified provider.</p>
                <p className={styles.note}>Note: Sensitive headers are not shown.</p>
              </div>
              <div className={styles['code-output-container']}>
                <pre className={styles['code-output']}>{transformedRequest}</pre>
                <button className={styles['copy-button']} onClick={handleCopy}>
                  Copy
                </button>
              </div>
            </div>
          </>
        );
    }
  };

  return (
    <div className={styles['transform-playground']}>
      <div className={styles['view-toggle']}>
        <button
          className={viewMode === 'split' ? styles.active : ''}
          onClick={() => setViewMode('split')}
        >
          Split View
        </button>
        <button
          className={viewMode === 'request' ? styles.active : ''}
          onClick={() => setViewMode('request')}
        >
          Request
        </button>
        <button
          className={viewMode === 'transformed' ? styles.active : ''}
          onClick={() => setViewMode('transformed')}
        >
          Transformed
        </button>
      </div>
      <div className={styles['playground-container']}>
        {renderContent()}
      </div>
    </div>
  );
};

export default TransformRequestPlayground; 