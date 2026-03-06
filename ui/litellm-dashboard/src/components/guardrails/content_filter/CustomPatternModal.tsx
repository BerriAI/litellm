import React from "react";
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
  return (
    <Modal
      title="Add custom regex pattern"
      open={visible}
      onCancel={onCancel}
      footer={null}
      width={800}
    >
      <Space direction="vertical" style={{ width: "100%" }} size="large">
        <div>
          <Text strong>Pattern name</Text>
          <Input
            placeholder="e.g., internal_id, employee_code"
            value={patternName}
            onChange={(e) => onNameChange(e.target.value)}
            style={{ marginTop: 8 }}
          />
        </div>

        <div>
          <Text strong>Regex pattern</Text>
          <Input
            placeholder="e.g., ID-[0-9]{6}"
            value={patternRegex}
            onChange={(e) => onRegexChange(e.target.value)}
            style={{ marginTop: 8 }}
          />
          <Text type="secondary" style={{ fontSize: 12 }}>
            Enter a valid regular expression to match sensitive data
          </Text>
        </div>

        <div>
          <Text strong>Action</Text>
          <Text type="secondary" style={{ display: "block", marginTop: 4, marginBottom: 8 }}>
            Choose what action the guardrail should take when this pattern is detected
          </Text>
          <Select
            value={patternAction}
            onChange={onActionChange}
            style={{ width: "100%" }}
          >
            <Option value="BLOCK">Block</Option>
            <Option value="MASK">Mask</Option>
          </Select>
        </div>
      </Space>

      <div style={{ display: "flex", justifyContent: "flex-end", gap: "8px", marginTop: "24px" }}>
        <Button onClick={onCancel}>
          Cancel
        </Button>
        <Button type="primary" onClick={onAdd}>
          Add
        </Button>
      </div>
    </Modal>
  );
};

export default CustomPatternModal;

