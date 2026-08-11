import React from "react";
import { useTranslation } from "react-i18next";
import { Typography, Select, Modal, Space, Button, Input } from "antd";

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
  const { t } = useTranslation("gateway");
  return (
    <Modal
      title={t("guardrailsPage.contentFilter.addCustomPattern")}
      open={visible}
      onCancel={onCancel}
      footer={null}
      width={800}
    >
      <Space direction="vertical" style={{ width: "100%" }} size="large">
        <div>
          <Text strong>{t("guardrailsPage.contentFilter.patternName")}</Text>
          <Input
            placeholder="e.g., internal_id, employee_code"
            value={patternName}
            onChange={(e) => onNameChange(e.target.value)}
            style={{ marginTop: 8 }}
          />
        </div>

        <div>
          <Text strong>{t("guardrailsPage.contentFilter.regexPattern")}</Text>
          <Input
            placeholder="e.g., ID-[0-9]{6}"
            value={patternRegex}
            onChange={(e) => onRegexChange(e.target.value)}
            style={{ marginTop: 8 }}
          />
          <Text type="secondary" style={{ fontSize: 12 }}>
            {t("guardrailsPage.contentFilter.regexHelp")}
          </Text>
        </div>

        <div>
          <Text strong>{t("guardrailsPage.contentFilter.action")}</Text>
          <Text type="secondary" style={{ display: "block", marginTop: 4, marginBottom: 8 }}>
            {t("guardrailsPage.contentFilter.patternActionHelp")}
          </Text>
          <Select value={patternAction} onChange={onActionChange} style={{ width: "100%" }}>
            <Option value="BLOCK">{t("guardrailsPage.contentFilter.block")}</Option>
            <Option value="MASK">{t("guardrailsPage.contentFilter.mask")}</Option>
          </Select>
        </div>
      </Space>

      <div style={{ display: "flex", justifyContent: "flex-end", gap: "8px", marginTop: "24px" }}>
        <Button onClick={onCancel}>{t("guardrailsPage.contentFilter.cancel")}</Button>
        <Button type="primary" onClick={onAdd}>
          {t("guardrailsPage.contentFilter.add")}
        </Button>
      </div>
    </Modal>
  );
};

export default CustomPatternModal;
