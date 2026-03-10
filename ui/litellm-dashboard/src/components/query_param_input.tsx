import React, { useState } from "react";
import { Button, Space } from "antd";
import { MinusCircleOutlined, PlusOutlined } from "@ant-design/icons";
import { TextInput } from "@tremor/react";

interface QueryParamInputProps {
  value?: Record<string, string>;
  onChange?: (value: Record<string, string>) => void;
}

const QueryParamInput: React.FC<QueryParamInputProps> = ({ value = {}, onChange }) => {
  const [pairs, setPairs] = useState<[string, string][]>(Object.entries(value));

  const handleAdd = () => {
    setPairs([...pairs, ["", ""]]);
  };

  const handleRemove = (index: number) => {
    const newPairs = pairs.filter((_, i) => i !== index);
    setPairs(newPairs);
    onChange?.(Object.fromEntries(newPairs));
  };

  const handleChange = (index: number, key: string, val: string) => {
    const newPairs = [...pairs];
    newPairs[index] = [key, val];
    setPairs(newPairs);
    onChange?.(Object.fromEntries(newPairs));
  };

  return (
    <div>
      {pairs.map(([key, val], index) => (
        <Space key={index} style={{ display: "flex", marginBottom: 8 }} align="center">
          <TextInput
            placeholder="Parameter Name (e.g., version)"
            value={key}
            onChange={(e) => handleChange(index, e.target.value, val)}
          />
          <TextInput
            placeholder="Parameter Value (e.g., v1)"
            value={val}
            onChange={(e) => handleChange(index, key, e.target.value)}
          />
          <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%" }}>
            <MinusCircleOutlined onClick={() => handleRemove(index)} style={{ cursor: "pointer" }} />
          </div>
        </Space>
      ))}
      <Button type="dashed" onClick={handleAdd} icon={<PlusOutlined />}>
        Add Query Parameter
      </Button>
    </div>
  );
};

export default QueryParamInput;