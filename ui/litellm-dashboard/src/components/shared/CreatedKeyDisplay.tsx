import React, { useState } from "react";
import { CopyToClipboard } from "react-copy-to-clipboard";
import { Button, Select } from "antd";
import MessageManager from "@/components/molecules/message_manager";
import { keyShareOnePasswordCall } from "@/components/networking";

interface CreatedKeyDisplayProps {
  apiKey: string;
  accessToken?: string;
}

const EXPIRY_OPTIONS = [
  { value: "OneHour", label: "1 hour" },
  { value: "OneDay", label: "1 day" },
  { value: "SevenDays", label: "7 days" },
  { value: "FourteenDays", label: "14 days" },
  { value: "ThirtyDays", label: "30 days" },
];

/**
 * Shared component for displaying a newly-created virtual key.
 * Used on the Virtual Keys page and in the Add Agent wizard.
 */
const CreatedKeyDisplay: React.FC<CreatedKeyDisplayProps> = ({ apiKey, accessToken }) => {
  const [copied, setCopied] = useState(false);
  const [sharing, setSharing] = useState(false);
  const [shareLink, setShareLink] = useState<string | null>(null);
  const [shareLinkCopied, setShareLinkCopied] = useState(false);
  const [expireAfter, setExpireAfter] = useState<string>("SevenDays");

  const handleCopy = () => {
    setCopied(true);
    MessageManager.success("Key copied to clipboard");
    setTimeout(() => setCopied(false), 2000);
  };

  const handleShareOnePassword = async () => {
    if (!accessToken) return;
    setSharing(true);
    try {
      const result = await keyShareOnePasswordCall(accessToken, apiKey, {
        expire_after: expireAfter,
      });
      setShareLink(result.share_link);
      MessageManager.success("1Password share link created");
    } catch (error) {
      MessageManager.error(
        `Failed to create 1Password share link: ${error instanceof Error ? error.message : String(error)}`,
      );
    } finally {
      setSharing(false);
    }
  };

  const handleShareLinkCopy = () => {
    setShareLinkCopied(true);
    MessageManager.success("Share link copied to clipboard");
    setTimeout(() => setShareLinkCopied(false), 2000);
  };

  return (
    <div>
      <p className="mb-2">
        Please save this secret key somewhere safe and accessible. For security reasons,{" "}
        <b>you will not be able to view it again</b> through your LiteLLM account. If you lose this secret key, you will
        need to generate a new one.
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
        <pre style={{ wordWrap: "break-word", whiteSpace: "normal", margin: 0 }}>{apiKey}</pre>
      </div>

      <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap", marginTop: 12 }}>
        <CopyToClipboard text={apiKey} onCopy={handleCopy}>
          <Button type="primary">{copied ? "Copied!" : "Copy Virtual Key"}</Button>
        </CopyToClipboard>

        {accessToken && !shareLink && (
          <>
            <Select
              value={expireAfter}
              onChange={setExpireAfter}
              options={EXPIRY_OPTIONS}
              style={{ width: 120 }}
              aria-label="Share link expiry"
            />
            <Button onClick={handleShareOnePassword} loading={sharing}>
              Share on 1Password
            </Button>
          </>
        )}
      </div>

      {shareLink && (
        <div style={{ marginTop: 16 }}>
          <p className="text-sm text-gray-600 mb-1">1Password secure share link:</p>
          <div
            style={{
              background: "#f8f8f8",
              padding: "10px",
              borderRadius: "5px",
              marginBottom: "10px",
            }}
          >
            <a href={shareLink} target="_blank" rel="noopener noreferrer" style={{ wordBreak: "break-all" }}>
              {shareLink}
            </a>
          </div>
          <CopyToClipboard text={shareLink} onCopy={handleShareLinkCopy}>
            <Button type="primary">{shareLinkCopied ? "Copied!" : "Copy Share Link"}</Button>
          </CopyToClipboard>
        </div>
      )}
    </div>
  );
};

export default CreatedKeyDisplay;
