import React, { useState } from 'react';
import { Form, Input, Button, Space } from 'antd';
import { MinusCircleOutlined, PlusOutlined } from '@ant-design/icons';
import { TextInput, Grid, Col } from "@tremor/react";
import { TrashIcon } from "@heroicons/react/outline";

interface KeyValueInputProps {
  value?: Record<string, string>;
  onChange?: (value: Record<string, string>) => void;
}

const KeyValueInput: React.FC<KeyValueInputProps> = ({ value = {}, onChange }) => {
  const [pairs, setPairs] = useState<[string, string][]>(Object.entries(value));

  const handleAdd = () => {
    setPairs([...pairs, ['', '']]);
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
        <Space key={index} style={{ display: 'flex', marginBottom: 8 }} align="start">
          <TextInput
            placeholder="Header Name"
            value={key}
            onChange={(e) => handleChange(index, e.target.value, val)}
          />
          <TextInput
            placeholder="Header Value"
            value={val}
            onChange={(e) => handleChange(index, key, e.target.value)}
          />
          <MinusCircleOutlined onClick={() => handleRemove(index)} />
        </Space>
      ))}
      <Button type="dashed" onClick={handleAdd} icon={<PlusOutlined />}>
        Add Header
      </Button>
    </div>
  );
};

export default KeyValueInput;
