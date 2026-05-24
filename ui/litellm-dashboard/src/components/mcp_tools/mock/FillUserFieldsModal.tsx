// Per-user env-vars fill modal — DB-backed (no longer a mock).
// Loads the per-user `MCPEnvVarDefinitionPublic` list + current values from
// `GET /v1/mcp/server/{server_id}/my-env-vars` and saves via POST. Lives in
// `mock/` for historical reasons; safe to relocate once we kill that folder.

import React, { useEffect, useMemo, useState } from "react";
import { Modal, Input, Form, Typography, Tag, Alert, Spin } from "antd";
import { Button } from "@tremor/react";
import {
  MCPEnvVarDefinitionPublic,
  MCPUserEnvVarsStatus,
  getMyMcpEnvVars,
  storeMyMcpEnvVars,
} from "../../networking";

const { Text, Title } = Typography;

interface FillUserFieldsModalProps {
  open: boolean;
  serverId: string;
  serverAlias: string;
  serverName?: string | null;
  accessToken: string;
  onClose: () => void;
  onSaved?: (status: MCPUserEnvVarsStatus) => void;
}

const FillUserFieldsModal: React.FC<FillUserFieldsModalProps> = ({
  open,
  serverId,
  serverAlias,
  serverName,
  accessToken,
  onClose,
  onSaved,
}) => {
  const [defs, setDefs] = useState<MCPEnvVarDefinitionPublic[]>([]);
  const [values, setValues] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);

  useEffect(() => {
    if (!open || !serverId || !accessToken) return;
    let cancelled = false;
    setLoading(true);
    setLoadError(null);
    setSaveError(null);
    getMyMcpEnvVars(accessToken, serverId)
      .then((status) => {
        if (cancelled) return;
        setDefs(status.definitions);
        setValues(status.values || {});
      })
      .catch((e: Error) => {
        if (cancelled) return;
        setLoadError(e.message || "Failed to load credentials");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [open, serverId, accessToken]);

  const perUserDefs = useMemo(
    () => defs.filter((d) => d.scope === "per_user"),
    [defs],
  );

  const handleSave = async () => {
    setSaving(true);
    setSaveError(null);
    try {
      const status = await storeMyMcpEnvVars(accessToken, serverId, values);
      onSaved?.(status);
      onClose();
    } catch (e: any) {
      setSaveError(e?.message || "Failed to save credentials");
    } finally {
      setSaving(false);
    }
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
            <Tag color="blue">Per-user</Tag>
          </div>
          <Text type="secondary" className="text-xs">
            {serverName || serverAlias}
          </Text>
        </div>
      }
      width={520}
    >
      <div className="space-y-4 mt-2">
        {loading ? (
          <div className="flex items-center justify-center py-8">
            <Spin />
          </div>
        ) : loadError ? (
          <Alert type="error" showIcon message={loadError} />
        ) : perUserDefs.length === 0 ? (
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
            {saveError && (
              <Alert type="error" showIcon message={saveError} />
            )}
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
