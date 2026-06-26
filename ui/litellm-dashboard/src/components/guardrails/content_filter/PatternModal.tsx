import React from "react";
import { Typography, Select, Modal, Space, Button } from "antd";
import { useTranslation } from "react-i18next";

const { Text } = Typography;
const { Option } = Select;

interface PrebuiltPattern {
  name: string;
  display_name: string;
  category: string;
  description: string;
}

interface PatternModalProps {
  visible: boolean;
  prebuiltPatterns: PrebuiltPattern[];
  categories: string[];
  selectedPatternName: string;
  patternAction: "BLOCK" | "MASK";
  onPatternNameChange: (name: string) => void;
  onActionChange: (action: "BLOCK" | "MASK") => void;
  onAdd: () => void;
  onCancel: () => void;
}

const PatternModal: React.FC<PatternModalProps> = ({
  visible,
  prebuiltPatterns,
  categories,
  selectedPatternName,
  patternAction,
  onPatternNameChange,
  onActionChange,
  onAdd,
  onCancel,
}) => {
  const { t } = useTranslation();

  return (
    <Modal title={t("guardrails.patternModal.title")} open={visible} onCancel={onCancel} footer={null} width={800}>
      <Space direction="vertical" style={{ width: "100%" }} size="large">
        <div>
          <Text strong>{t("guardrails.patternModal.patternTypeLabel")}</Text>
          <Select
            placeholder={t("guardrails.patternModal.patternTypePlaceholder")}
            value={selectedPatternName}
            onChange={onPatternNameChange}
            style={{ width: "100%", marginTop: 8 }}
            showSearch
            filterOption={(input, option) => {
              const pattern = prebuiltPatterns.find((p) => p.name === option?.value);
              if (pattern) {
                return (
                  pattern.display_name.toLowerCase().includes(input.toLowerCase()) ||
                  pattern.name.toLowerCase().includes(input.toLowerCase())
                );
              }
              return false;
            }}
          >
            {categories.map((category) => {
              const categoryPatterns = prebuiltPatterns.filter((p) => p.category === category);
              if (categoryPatterns.length === 0) return null;

              return (
                <Select.OptGroup key={category} label={category}>
                  {categoryPatterns.map((pattern) => (
                    <Option key={pattern.name} value={pattern.name}>
                      {pattern.display_name}
                    </Option>
                  ))}
                </Select.OptGroup>
              );
            })}
          </Select>
        </div>

        <div>
          <Text strong>{t("guardrails.patternModal.actionLabel")}</Text>
          <Text type="secondary" style={{ display: "block", marginTop: 4, marginBottom: 8 }}>
            {t("guardrails.patternModal.actionDesc")}
          </Text>
          <Select value={patternAction} onChange={onActionChange} style={{ width: "100%" }}>
            <Option value="BLOCK">{t("guardrails.patternModal.actionBlock")}</Option>
            <Option value="MASK">{t("guardrails.patternModal.actionMask")}</Option>
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

export default PatternModal;
