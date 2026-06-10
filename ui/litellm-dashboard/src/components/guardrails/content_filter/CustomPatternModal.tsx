import React from "react";
import { Typography, Select, Modal, Space, Button, Input } from "antd";
import { useTranslation } from "react-i18next";

const { Text } = Typography;
const { Option } = Select;

interface CustomPatternModalProps {
  visible: boolean;
  patternName: string;
  patternRegex: string;
  patternAction: "BLOCK" | "MASK";
  onNameChange: (name: string) => void;
  onRegexChange: (regex: string) => void;
  onActionChange: (action: "BLOCK" | "MASK") => void;
  onAdd: () => void;
  onCancel: () => void;
}

const CustomPatternModal: React.FC<CustomPatternModalProps> = ({
  visible,
  patternName,
  patternRegex,
  patternAction,
  onNameChange,
  onRegexChange,
  onActionChange,
  onAdd,
  onCancel,
}) => {
  const { t } = useTranslation();
  return (
    <Modal
      title={t("guardrails.customPatternModal.title")}
      open={visible}
      onCancel={onCancel}
      footer={null}
      width={800}
    >
      <Space direction="vertical" style={{ width: "100%" }} size="large">
        <div>
          <Text strong>{t("guardrails.customPatternModal.patternNameLabel")}</Text>
          <Input
            placeholder={t("guardrails.customPatternModal.patternNamePlaceholder")}
            value={patternName}
            onChange={(e) => onNameChange(e.target.value)}
            style={{ marginTop: 8 }}
          />
        </div>

        <div>
          <Text strong>{t("guardrails.customPatternModal.regexPatternLabel")}</Text>
          <Input
            placeholder={t("guardrails.customPatternModal.regexPatternPlaceholder")}
            value={patternRegex}
            onChange={(e) => onRegexChange(e.target.value)}
            style={{ marginTop: 8 }}
          />
          <Text type="secondary" style={{ fontSize: 12 }}>
            {t("guardrails.customPatternModal.regexPatternHelp")}
          </Text>
        </div>

        <div>
          <Text strong>{t("guardrails.customPatternModal.actionLabel")}</Text>
          <Text type="secondary" style={{ display: "block", marginTop: 4, marginBottom: 8 }}>
            {t("guardrails.customPatternModal.actionHelp")}
          </Text>
          <Select value={patternAction} onChange={onActionChange} style={{ width: "100%" }}>
            <Option value="BLOCK">{t("guardrails.customPatternModal.actionBlock")}</Option>
            <Option value="MASK">{t("guardrails.customPatternModal.actionMask")}</Option>
          </Select>
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

export default CustomPatternModal;
