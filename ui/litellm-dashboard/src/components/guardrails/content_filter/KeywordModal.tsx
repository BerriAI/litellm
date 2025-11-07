import React from "react";
import { Typography, Select, Modal, Space } from "antd";
import { Button, TextInput, Textarea } from "@tremor/react";

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
  return (
    <Modal
      title="Add blocked keyword"
      open={visible}
      onCancel={onCancel}
      footer={null}
      width={800}
    >
      <Space direction="vertical" style={{ width: "100%" }} size="large">
        <div>
          <Text strong>Keyword</Text>
          <TextInput
            placeholder="Enter sensitive keyword or phrase"
            value={keyword}
            onValueChange={onKeywordChange}
            style={{ marginTop: 8 }}
          />
        </div>

        <div>
          <Text strong>Action</Text>
          <Text type="secondary" style={{ display: "block", marginTop: 4, marginBottom: 8 }}>
            Choose what action the guardrail should take when this keyword is detected
          </Text>
          <Select
            value={action}
            onChange={onActionChange}
            style={{ width: "100%" }}
          >
            <Option value="BLOCK">Block</Option>
            <Option value="MASK">Mask</Option>
          </Select>
        </div>

        <div>
          <Text strong>Description (optional)</Text>
          <Textarea
            placeholder="Explain why this keyword is sensitive"
            value={description}
            onValueChange={onDescriptionChange}
            rows={3}
            style={{ marginTop: 8 }}
          />
        </div>
      </Space>

      <div style={{ display: "flex", justifyContent: "flex-end", gap: "8px", marginTop: "24px" }}>
        <Button variant="secondary" onClick={onCancel}>
          Cancel
        </Button>
        <Button onClick={onAdd}>
          Add
        </Button>
      </div>
    </Modal>
  );
};

export default KeywordModal;

