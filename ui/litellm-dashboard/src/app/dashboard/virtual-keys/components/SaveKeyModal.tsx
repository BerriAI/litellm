"use client";

import { Button, Col, Grid, Text, Title } from "@tremor/react";
import { CopyToClipboard } from "react-copy-to-clipboard";
import { Modal } from "antd";
import React from "react";
import NotificationsManager from "@/components/molecules/notifications_manager";

export interface SaveKeyModalProps {
  apiKey: string;
  isModalVisible: boolean;
  handleOk: () => void;
  handleCancel: () => void;
}

const SaveKeyModal = ({ apiKey, isModalVisible, handleOk, handleCancel }: SaveKeyModalProps) => {
  const handleCopy = () => {
    NotificationsManager.success("API Key copied to clipboard");
  };

  return (
    <Modal open={isModalVisible} onOk={handleOk} onCancel={handleCancel} footer={null}>
      <Grid numItems={1} className="gap-2 w-full">
        <Title>Save your Key</Title>
        <Col numColSpan={1}>
          <p>
            Please save this secret key somewhere safe and accessible. For security reasons,{" "}
            <b>you will not be able to view it again</b> through your LiteLLM account. If you lose this secret key, you
            will need to generate a new one.
          </p>
        </Col>
        <Col numColSpan={1}>
          {apiKey != null ? (
            <div>
              <Text className="mt-3">API Key:</Text>
              <div
                style={{
                  background: "#f8f8f8",
                  padding: "10px",
                  borderRadius: "5px",
                  marginBottom: "10px",
                }}
              >
                <pre style={{ wordWrap: "break-word", whiteSpace: "normal" }}>{apiKey}</pre>
              </div>

              <CopyToClipboard text={apiKey} onCopy={handleCopy}>
                <Button className="mt-3">Copy API Key</Button>
              </CopyToClipboard>
            </div>
          ) : (
            <Text>Key being created, this might take 30s</Text>
          )}
        </Col>
      </Grid>
    </Modal>
  );
};

export default SaveKeyModal;
