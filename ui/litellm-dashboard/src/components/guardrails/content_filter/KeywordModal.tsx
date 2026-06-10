import React from "react";
import { Typography, Select, Modal, Space, Button, Input } from "antd";
import { useTranslation } from "react-i18next";

const { Text } = Typography;
const { Option } = Select;

interface KeywordModalProps {
  visible: boolean;
  keyword: string;
  action: "BLOCK" | "MASK";
  description: string;
  onKeywordChange: (keyword: string) => void;
  onActionChange: (action: "BLOCK" | "MASK") => void;
  onDescriptionChange: (description: string) => void;
  onAdd: () => void;
  onCancel: () => void;
}

const KeywordModal: React.FC<KeywordModalProps> = ({
  visible,
  keyword,
  action,
  description,
  onKeywordChange,
  onActionChange,
  onDescriptionChange,
  onAdd,
  onCancel,
}) => {
  const { t } = useTranslation();

  return (
    <Modal title={t("guardrails.keywordModal.title")} open={visible} onCancel={onCancel} footer={null} width={800}>
      <Space direction="vertical" style={{ width: "100%" }} size="large">
        <div>
          <Text strong>{t("guardrails.keywordModal.keywordLabel")}</Text>
          <Input
            placeholder={t("guardrails.keywordModal.keywordPlaceholder")}
            value={keyword}
            onChange={(e) => onKeywordChange(e.target.value)}
            style={{ marginTop: 8 }}
          />
        </div>

        <div>
          <Text strong>{t("guardrails.keywordModal.actionLabel")}</Text>
          <Text type="secondary" style={{ display: "block", marginTop: 4, marginBottom: 8 }}>
            {t("guardrails.keywordModal.actionDesc")}
          </Text>
          <Select value={action} onChange={onActionChange} style={{ width: "100%" }}>
            <Option value="BLOCK">{t("guardrails.keywordModal.actionBlock")}</Option>
            <Option value="MASK">{t("guardrails.keywordModal.actionMask")}</Option>
          </Select>
        </div>

        <div>
          <Text strong>{t("guardrails.keywordModal.descriptionLabel")}</Text>
          <Input.TextArea
            placeholder={t("guardrails.keywordModal.descriptionPlaceholder")}
            value={description}
            onChange={(e) => onDescriptionChange(e.target.value)}
            rows={3}
            style={{ marginTop: 8 }}
          />
        </div>
      </Space>

      <div style={{ display: "flex", justifyContent: "flex-end", gap: "8px", marginTop: "24px" }}>
        <Button onClick={onCancel}>{t("common.cancel")}</Button>
        <Button type="primary" onClick={onAdd}>
          {t("common.add")}
        </Button>
      </div>
    </Modal>
  );
};

export default KeywordModal;
