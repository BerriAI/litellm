import React, { useState } from "react";
import { CopyToClipboard } from "react-copy-to-clipboard";
import { Button } from "antd";
import { Trans, useTranslation } from "react-i18next";
import MessageManager from "@/components/molecules/message_manager";

interface CreatedKeyDisplayProps {
  apiKey: string;
}

const CreatedKeyDisplay: React.FC<CreatedKeyDisplayProps> = ({ apiKey }) => {
  const { t } = useTranslation();
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    setCopied(true);
    MessageManager.success(t("shared.createdKeyDisplay.copiedToClipboard"));
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div>
      <p className="mb-2">
        <Trans i18nKey="shared.createdKeyDisplay.saveWarning" components={{ b: <b /> }} />
      </p>

      <p className="text-sm text-gray-600 mt-3 mb-1">{t("shared.createdKeyDisplay.virtualKeyLabel")}</p>
      <div
        style={{
          background: "#f8f8f8",
          padding: "10px",
          borderRadius: "5px",
          marginBottom: "10px",
        }}
      >
        <pre style={{ wordWrap: "break-word", whiteSpace: "normal", margin: 0 }}>{apiKey}</pre>
      </div>

      <CopyToClipboard text={apiKey} onCopy={handleCopy}>
        <Button type="primary" style={{ marginTop: 12 }}>
          {copied ? t("common.copied") : t("shared.createdKeyDisplay.copyButton")}
        </Button>
      </CopyToClipboard>
    </div>
  );
};

export default CreatedKeyDisplay;
