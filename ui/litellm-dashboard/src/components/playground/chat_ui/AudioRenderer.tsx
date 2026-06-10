import React from "react";
import { useTranslation } from "react-i18next";
import { MessageType } from "./types";

interface AudioRendererProps {
  message: MessageType;
}

const AudioRenderer: React.FC<AudioRendererProps> = ({ message }) => {
  const { t } = useTranslation();
  // Check if this message contains audio
  if (!message.isAudio || typeof message.content !== "string") {
    return null;
  }

  return (
    <div className="mb-2">
      <audio controls src={message.content} className="max-w-full" style={{ maxWidth: "500px" }}>
        {t("playground.audioRenderer.browserNotSupported")}
      </audio>
    </div>
  );
};

export default AudioRenderer;
