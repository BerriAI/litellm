import React, { useState } from "react";
import { CopyToClipboard } from "react-copy-to-clipboard";
import { Button } from "@/components/ui/button";
import MessageManager from "@/components/molecules/message_manager";

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
    MessageManager.success("Key copied to clipboard");
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div>
      <p className="mb-2">
        Please save this secret key somewhere safe and accessible. For security
        reasons, <b>you will not be able to view it again</b> through your
        LiteLLM account. If you lose this secret key, you will need to generate
        a new one.
      </p>

      <p className="text-sm text-muted-foreground mt-3 mb-1">Virtual Key:</p>
      <div className="bg-muted p-2.5 rounded-md mb-2.5">
        <pre className="break-words whitespace-normal m-0 text-sm">
          {apiKey}
        </pre>
      </div>

      <CopyToClipboard text={apiKey} onCopy={handleCopy}>
        <Button className="mt-3">
          {copied ? "Copied!" : "Copy Virtual Key"}
        </Button>
      </CopyToClipboard>
    </div>
  );
};

export default CreatedKeyDisplay;
