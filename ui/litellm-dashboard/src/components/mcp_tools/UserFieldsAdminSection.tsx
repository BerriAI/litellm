import React from "react";
import { Input, Switch, Typography, Tooltip } from "antd";
import { InfoCircleOutlined, PlusOutlined, DeleteOutlined } from "@ant-design/icons";
import { UserField } from "./userFields";

const { Title, Paragraph, Text } = Typography;

interface UserFieldsAdminSectionProps {
  value: UserField[];
  onChange: (next: UserField[]) => void;
}

const blankField = (): UserField => ({
  name: "",
  label: "",
  description: "",
  secret: true,
});

const UserFieldsAdminSection: React.FC<UserFieldsAdminSectionProps> = ({ value, onChange }) => {
  const fields = value || [];

  const update = (idx: number, patch: Partial<UserField>) => {
    onChange(fields.map((f, i) => (i === idx ? { ...f, ...patch } : f)));
  };

  const remove = (idx: number) => {
    onChange(fields.filter((_, i) => i !== idx));
  };

  const add = () => {
    onChange([...fields, blankField()]);
  };

  return (
    <div className="border border-gray-200 rounded-lg p-4 bg-gray-50">
      <div className="flex items-start justify-between mb-3">
        <div>
          <Title level={5} style={{ margin: 0 }}>
            Per-User Fields
            <Tooltip title="Fields each end-user must fill in themselves before using this MCP server. Use for per-user secrets like personal bearer tokens, account IDs, etc.">
              <InfoCircleOutlined className="ml-2 text-blue-400" />
            </Tooltip>
          </Title>
          <Paragraph type="secondary" style={{ margin: 0, fontSize: 12 }}>
            Each user will be prompted to fill these in on their dashboard. The MCP server will refuse
            connections until they do.
          </Paragraph>
        </div>
        <button
          type="button"
          onClick={add}
          className="text-sm bg-blue-600 hover:bg-blue-700 text-white px-3 py-1 rounded-md font-medium transition-colors flex items-center gap-1"
        >
          <PlusOutlined /> Add Field
        </button>
      </div>

      {fields.length === 0 ? (
        <div className="text-center py-6 text-gray-400 text-sm bg-white rounded border border-dashed border-gray-300">
          No per-user fields. Click <strong>Add Field</strong> to require users to provide their own
          values (e.g. a personal bearer token).
        </div>
      ) : (
        <div className="space-y-3">
          {fields.map((f, idx) => (
            <div key={f.name || idx} className="bg-white rounded border border-gray-200 p-3">
              <div className="grid grid-cols-12 gap-2 items-start">
                <div className="col-span-3">
                  <Text type="secondary" style={{ fontSize: 11 }}>Field Name (key)</Text>
                  <Input
                    placeholder="bearer_token"
                    value={f.name}
                    onChange={(e) => update(idx, { name: e.target.value })}
                    size="middle"
                  />
                </div>
                <div className="col-span-3">
                  <Text type="secondary" style={{ fontSize: 11 }}>Label</Text>
                  <Input
                    placeholder="Personal Gmail Token"
                    value={f.label}
                    onChange={(e) => update(idx, { label: e.target.value })}
                    size="middle"
                  />
                </div>
                <div className="col-span-4">
                  <Text type="secondary" style={{ fontSize: 11 }}>Description (shown to user)</Text>
                  <Input
                    placeholder="Generate one at https://..."
                    value={f.description}
                    onChange={(e) => update(idx, { description: e.target.value })}
                    size="middle"
                  />
                </div>
                <div className="col-span-1 pt-4">
                  <Tooltip title="Treat as secret (masked input)">
                    <Switch
                      checked={!!f.secret}
                      onChange={(checked) => update(idx, { secret: checked })}
                      size="small"
                    />
                  </Tooltip>
                </div>
                <div className="col-span-1 pt-3 text-right">
                  <Tooltip title="Remove field">
                    <button
                      type="button"
                      onClick={() => remove(idx)}
                      className="p-1 text-gray-400 hover:text-red-600"
                    >
                      <DeleteOutlined />
                    </button>
                  </Tooltip>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default UserFieldsAdminSection;
