// PROTOTYPE: modal where an end-user fills in their per-user fields for an
// MCP server (mock — values stored in localStorage keyed by user + server alias).

import React, { useEffect, useMemo, useState } from "react";
import { Modal, Input, Form, Typography, Tag, Alert } from "antd";
import { Button } from "@tremor/react";
import {
  EnvVarDefinition,
  getEnvVarDefinitions,
  getPerUserValues,
  setPerUserValues,
  notifyEnvVarsChanged,
} from "./mockMcpEnvVars";

const { Text, Title } = Typography;

interface FillUserFieldsModalProps {
  open: boolean;
  serverAlias: string;
  serverName?: string | null;
  userId: string;
  onClose: () => void;
  onSaved?: () => void;
}

const FillUserFieldsModal: React.FC<FillUserFieldsModalProps> = ({
  open,
  serverAlias,
  serverName,
  userId,
  onClose,
  onSaved,
}) => {
  const [defs, setDefs] = useState<EnvVarDefinition[]>([]);
  const [values, setValues] = useState<Record<string, string>>({});
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (!open || !serverAlias) return;
    const loadedDefs = getEnvVarDefinitions(serverAlias);
    setDefs(loadedDefs);
    setValues(getPerUserValues(serverAlias, userId));
  }, [open, serverAlias, userId]);

  const perUserDefs = useMemo(
    () => defs.filter((d) => d.scope === "per_user"),
    [defs],
  );

  const handleSave = () => {
    setSaving(true);
    // Simulate brief latency so the demo feels real.
    setTimeout(() => {
      setPerUserValues(serverAlias, userId, values);
      notifyEnvVarsChanged();
      setSaving(false);
      onSaved?.();
      onClose();
    }, 250);
  };

  return (
    <Modal
      open={open}
      onCancel={onClose}
      footer={null}
      title={
        <div>
          <div className="flex items-center gap-2">
            <Title level={5} style={{ margin: 0 }}>
              Set your credentials
            </Title>
            <Tag color="purple">Prototype</Tag>
          </div>
          <Text type="secondary" className="text-xs">
            {serverName || serverAlias}
          </Text>
        </div>
      }
      width={520}
    >
      <div className="space-y-4 mt-2">
        {perUserDefs.length === 0 ? (
          <Alert
            type="info"
            showIcon
            message="No per-user fields configured for this server."
          />
        ) : (
          <>
            <Text className="text-sm text-gray-600 block">
              These values are private to you. Your admin configured this MCP
              server to require these per-user credentials:
            </Text>
            <Form layout="vertical">
              {perUserDefs.map((d) => (
                <Form.Item
                  key={d.name}
                  label={
                    <span className="font-mono text-sm font-semibold">
                      {d.name}
                    </span>
                  }
                  required
                >
                  <Input.Password
                    value={values[d.name] ?? ""}
                    onChange={(e) =>
                      setValues((prev) => ({
                        ...prev,
                        [d.name]: e.target.value,
                      }))
                    }
                    placeholder={`Enter your ${d.name}`}
                    visibilityToggle
                  />
                </Form.Item>
              ))}
            </Form>
            <div className="flex items-center justify-end gap-2 pt-2 border-t border-gray-100">
              <Button variant="secondary" onClick={onClose}>
                Cancel
              </Button>
              <Button
                variant="primary"
                onClick={handleSave}
                loading={saving}
                disabled={perUserDefs.some(
                  (d) => !values[d.name] || values[d.name].trim() === "",
                )}
              >
                Save Credentials
              </Button>
            </div>
          </>
        )}
      </div>
    </Modal>
  );
};

export default FillUserFieldsModal;
