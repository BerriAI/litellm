import React, { useState } from "react";
import { CopyToClipboard } from "react-copy-to-clipboard";
import { Button, message } from "antd";

interface CreatedKeyDisplayProps {
  apiKey: string;
}

/**
 * Shared component for displaying a newly-created virtual key.
 * Used on the Virtual Keys page and in the Add Agent wizard.
 */
const CreatedKeyDisplay: React.FC<CreatedKeyDisplayProps> = ({ apiKey }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    setCopied(true);
    message.success("Key copied to clipboard");
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div>
      <p className="mb-2">
        Please save this secret key somewhere safe and accessible. For security reasons,{" "}
        <b>you will not be able to view it again</b> through your LiteLLM account. If you
        lose this secret key, you will need to generate a new one.
      </p>

      <p className="text-sm text-gray-600 mt-3 mb-1">Virtual Key:</p>
      <div
        style={{
          background: "#f8f8f8",
          padding: "10px",
          borderRadius: "5px",
          marginBottom: "10px",
        }}
      >
        <pre style={{ wordWrap: "break-word", whiteSpace: "normal", margin: 0 }}>
          {apiKey}
        </pre>
      </div>

      <CopyToClipboard text={apiKey} onCopy={handleCopy}>
        <Button type="primary" style={{ marginTop: 12 }}>
          {copied ? "Copied!" : "Copy Virtual Key"}
        </Button>
      </CopyToClipboard>
    </div>
  );
};

export default CreatedKeyDisplay;
