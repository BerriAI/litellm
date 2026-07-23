import React, { useState } from "react";
import { CopyToClipboard } from "react-copy-to-clipboard";
import { Button } from "antd";
import MessageManager from "@/components/molecules/message_manager";
import { keyShareCreateCall } from "@/components/networking";

interface CreatedKeyDisplayProps {
  apiKey: string;
  accessToken?: string;
}

const PASSWORD_LINK_LOGO = "/ui/assets/logos/password_link_white.svg";
const PASSWORD_LINK_PURPLE = "#65428F";

/**
 * Shared component for displaying a newly-created virtual key.
 * Used on the Virtual Keys page and in the Add Agent wizard.
 */
const CreatedKeyDisplay: React.FC<CreatedKeyDisplayProps> = ({ apiKey, accessToken }) => {
  const [copied, setCopied] = useState(false);
  const [linkCopied, setLinkCopied] = useState(false);
  const [sharing, setSharing] = useState(false);
  const [shareLink, setShareLink] = useState<string | null>(null);

  const handleCopy = () => {
    setCopied(true);
    MessageManager.success("Key copied to clipboard");
    setTimeout(() => setCopied(false), 2000);
  };

  const handleLinkCopy = () => {
    setLinkCopied(true);
    MessageManager.success("Share link copied to clipboard");
    setTimeout(() => setLinkCopied(false), 2000);
  };

  const handleShare = async () => {
    if (!accessToken) return;
    setSharing(true);
    try {
      const response = await keyShareCreateCall(accessToken, apiKey);
      setShareLink(response.share_link);
      MessageManager.success("Secure share link created");
    } catch (error) {
      const detail = error instanceof Error ? error.message : "Please try again.";
      MessageManager.error(`Failed to create secure share link. ${detail}`);
    } finally {
      setSharing(false);
    }
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

      <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
        <CopyToClipboard text={apiKey} onCopy={handleCopy}>
          <Button type="primary" style={{ marginTop: 12 }}>
            {copied ? "Copied!" : "Copy Virtual Key"}
          </Button>
        </CopyToClipboard>

        {accessToken && (
          <Button
            onClick={handleShare}
            loading={sharing}
            style={{
              marginTop: 12,
              background: PASSWORD_LINK_PURPLE,
              borderColor: PASSWORD_LINK_PURPLE,
              color: "#ffffff",
              display: "flex",
              alignItems: "center",
              gap: 8,
            }}
          >
            <span>Securely share with</span>
            <span
              role="img"
              aria-label="Password.link"
              style={{
                display: "inline-block",
                width: 86,
                height: 14,
                backgroundImage: `url(${PASSWORD_LINK_LOGO})`,
                backgroundRepeat: "no-repeat",
                backgroundPosition: "center",
                backgroundSize: "contain",
              }}
            />
          </Button>
        )}
      </div>

      {shareLink && (
        <div style={{ marginTop: 16 }}>
          <p className="text-sm text-gray-600 mb-1">
            One-time secure link (reveals the key once, then self-destructs). Send it to the recipient:
          </p>
          <div
            style={{
              background: "#f8f8f8",
              padding: "10px",
              borderRadius: "5px",
              marginBottom: "10px",
            }}
          >
            <pre style={{ wordWrap: "break-word", whiteSpace: "normal", margin: 0 }}>{shareLink}</pre>
          </div>
          <CopyToClipboard text={shareLink} onCopy={handleLinkCopy}>
            <Button>{linkCopied ? "Copied!" : "Copy Share Link"}</Button>
          </CopyToClipboard>
        </div>
      )}
    </div>
  );
};

export default CreatedKeyDisplay;
