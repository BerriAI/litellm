// PROTOTYPE: global per-user variables tab.
//
// Behaviour notes (mock implementation):
//   - Per-user variables are global, *not* scoped to any MCP instance. They
//     are matched to template/instance variables by name at runtime.
//   - Values are write-only from the UI's POV — once saved they are pushed
//     to the configured credential store (HashiCorp Vault, AWS Secrets,
//     LiteLLM DB, …) keyed by user id, encrypted opaque blob.
//   - On reload the UI doesn't know the value or its length, so it renders
//     8 dots and disables the visibility toggle until the field is replaced.
//   - "Reveal" only works on values the user has just typed in this session.
//   - Saves are batched into one blob per user; admins never see plaintext.

import React, { useEffect, useMemo, useState } from "react";
import {
  Alert,
  Button,
  Card,
  Form,
  Input,
  Modal,
  Popconfirm,
  Select,
  Space,
  Switch,
  Tag,
  Tooltip,
  Typography,
} from "antd";
import {
  EyeInvisibleOutlined,
  EyeOutlined,
  InfoCircleOutlined,
  LockOutlined,
  PlusOutlined,
  ReloadOutlined,
  SaveOutlined,
} from "@ant-design/icons";
import {
  CREDENTIAL_STORE_LABELS,
  CredentialStoreConfig,
  CredentialStoreProvider,
  UserVariableEntry,
  getCredentialStoreConfig,
  getRevealedSet,
  listUserVariables,
  markVariableRevealed,
  saveCredentialStoreConfig,
  saveUserVariables,
  simulateReload,
  subscribeUserVariablesChanged,
} from "./mockUserVariables";

const { Title, Text, Paragraph } = Typography;

interface VariablesTabProps {
  userId: string;
  isAdmin: boolean;
}

// What a row looks like in the editor. `stored = true` means the credential
// store has a value for this name but we don't know it locally.
interface RowState {
  name: string;
  // Empty string + stored=true means "8 mask dots, hidden, can't reveal".
  value: string;
  stored: boolean;
  // True iff the user has typed a new value in this session — gates the
  // visibility toggle and the actual save payload.
  modifiedInSession: boolean;
  // Show plaintext or mask. Only honored if modifiedInSession=true.
  visible: boolean;
  updated_at?: string;
}

const PLACEHOLDER_DOTS = "••••••••";

const VariablesTab: React.FC<VariablesTabProps> = ({ userId, isAdmin }) => {
  const [storeCfg, setStoreCfg] = useState<CredentialStoreConfig>(() =>
    getCredentialStoreConfig(),
  );
  const [storeModalOpen, setStoreModalOpen] = useState(false);
  const [rows, setRows] = useState<RowState[]>([]);
  const [saving, setSaving] = useState(false);
  const [savedAt, setSavedAt] = useState<string | null>(null);

  // hydrate from "credential store"
  const hydrate = () => {
    const persisted = listUserVariables(userId);
    const revealed = new Set(getRevealedSet(userId));
    setRows(
      persisted.map((v) => ({
        name: v.name,
        // If the user revealed/edited it this session, we still have the
        // plaintext in localStorage (write-store sim). Otherwise pretend we
        // don't know it.
        value: revealed.has(v.name) ? v.value : "",
        stored: true,
        modifiedInSession: false,
        visible: false,
        updated_at: v.updated_at,
      })),
    );
    setStoreCfg(getCredentialStoreConfig());
  };

  useEffect(() => {
    hydrate();
    return subscribeUserVariablesChanged(hydrate);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [userId]);

  const isDirty = rows.some((r) => r.modifiedInSession);

  const handleNameChange = (idx: number, name: string) => {
    setRows((prev) =>
      prev.map((r, i) => (i === idx ? { ...r, name } : r)),
    );
  };

  const handleValueChange = (idx: number, value: string) => {
    setRows((prev) =>
      prev.map((r, i) =>
        i === idx
          ? {
              ...r,
              value,
              modifiedInSession: true,
              // Newly-typed values start hidden — user opts in to reveal.
              visible: r.visible,
            }
          : r,
      ),
    );
  };

  const toggleVisible = (idx: number) => {
    setRows((prev) =>
      prev.map((r, i) =>
        i === idx && r.modifiedInSession ? { ...r, visible: !r.visible } : r,
      ),
    );
  };

  const addRow = () => {
    setRows((prev) => [
      ...prev,
      {
        name: "",
        value: "",
        stored: false,
        modifiedInSession: true,
        visible: false,
      },
    ]);
  };

  const removeRow = (idx: number) => {
    setRows((prev) => prev.filter((_, i) => i !== idx));
  };

  const handleSave = () => {
    setSaving(true);
    const now = new Date().toISOString();
    const next = rows
      .filter((r) => r.name.trim() !== "")
      .map<UserVariableEntry>((r) => ({
        name: r.name.trim(),
        // In a real impl we'd send the new plaintext to the credential store
        // and keep nothing locally. Here we keep the value for the demo so the
        // simulated reload can put us back in the "unknown plaintext" state.
        value: r.modifiedInSession ? r.value : "",
        updated_at: r.modifiedInSession ? now : r.updated_at ?? now,
      }));
    // Pretend POST to credential store …
    setTimeout(() => {
      saveUserVariables(userId, next);
      // Mark each modified variable as "we still know plaintext in this tab"
      // until the user hits Simulate Reload.
      next.forEach((entry) => {
        const row = rows.find(
          (r) => r.name.trim() === entry.name && r.modifiedInSession,
        );
        if (row) markVariableRevealed(userId, entry.name);
      });
      setSavedAt(now);
      setSaving(false);
      hydrate();
    }, 350);
  };

  return (
    <div className="w-full">
      <div className="flex items-start justify-between mb-4">
        <div>
          <div className="flex items-center gap-3">
            <Title level={4} style={{ margin: 0 }}>
              Variables
            </Title>
            <Tag color="purple">Prototype</Tag>
          </div>
          <Text type="secondary" className="text-sm">
            Your personal values for per-user variables across all MCP servers.
            Matched to template variables by name at runtime.
          </Text>
        </div>
        <Space>
          <Tooltip title="Simulate a page refresh — the UI loses the plaintext for already-saved values, just like a real reload from the credential store.">
            <Button
              icon={<ReloadOutlined />}
              onClick={() => simulateReload(userId)}
            >
              Simulate Reload
            </Button>
          </Tooltip>
        </Space>
      </div>

      {/* Credential store backend */}
      <Card className="mb-4 border border-gray-200">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <LockOutlined className="text-purple-500 text-xl" />
            <div>
              <Text strong className="block text-sm">
                Credential Store Backend
              </Text>
              <Text type="secondary" className="text-xs">
                Where per-user variables are persisted. Configured by admin in{" "}
                <code>config.yaml</code> or via environment variables.
              </Text>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <Tag
              color={
                storeCfg.provider === "litellm_db" ? "default" : "purple"
              }
              style={{ marginRight: 0 }}
            >
              {CREDENTIAL_STORE_LABELS[storeCfg.provider]}
            </Tag>
            {isAdmin && (
              <Button size="small" onClick={() => setStoreModalOpen(true)}>
                Configure
              </Button>
            )}
          </div>
        </div>
        {storeCfg.provider === "hashicorp_vault" && storeCfg.hashicorp && (
          <div className="mt-3 pt-3 border-t border-gray-100 grid grid-cols-2 gap-x-6 gap-y-1 text-xs">
            <KV k="Vault Address" v={storeCfg.hashicorp.address} />
            <KV k="Mount Path" v={storeCfg.hashicorp.mount_path} />
            {storeCfg.hashicorp.namespace && (
              <KV k="Namespace" v={storeCfg.hashicorp.namespace} />
            )}
            <KV k="Auth Method" v={storeCfg.hashicorp.auth_method} />
          </div>
        )}
        {storeCfg.provider === "aws_secrets_manager" && storeCfg.aws && (
          <div className="mt-3 pt-3 border-t border-gray-100 grid grid-cols-2 gap-x-6 gap-y-1 text-xs">
            <KV k="Region" v={storeCfg.aws.region} />
            <KV k="Secret Prefix" v={storeCfg.aws.prefix} />
          </div>
        )}
      </Card>

      <Alert
        type="info"
        showIcon
        className="mb-4"
        message="Values are write-only"
        description={
          <span className="text-sm">
            Once saved, secrets are encrypted and pushed to the credential
            store. The UI can&apos;t read them back — that&apos;s why the
            password field shows dots of unknown length until you replace it.
            Try <b>Simulate Reload</b> after saving to see this in action.
          </span>
        }
      />

      <Card className="border border-gray-200">
        <div className="space-y-2">
          {rows.length === 0 ? (
            <div className="text-center py-8 text-gray-500 text-sm">
              No variables yet. Add one to get started — it will be available
              to every MCP server that declares a matching{" "}
              <code className="bg-gray-100 px-1 rounded">per_user</code>{" "}
              variable name.
            </div>
          ) : (
            <>
              <div className="flex gap-3 px-1 text-xs font-medium text-gray-500 uppercase tracking-wide">
                <div style={{ flex: 1 }}>Variable Name</div>
                <div style={{ flex: 1.4 }}>Value</div>
                <div style={{ width: 130 }}>Status</div>
                <div style={{ width: 32 }} />
              </div>
              {rows.map((row, idx) => (
                <VariableRow
                  key={idx}
                  row={row}
                  onNameChange={(v) => handleNameChange(idx, v)}
                  onValueChange={(v) => handleValueChange(idx, v)}
                  onToggleVisible={() => toggleVisible(idx)}
                  onRemove={() => removeRow(idx)}
                />
              ))}
            </>
          )}
          <Button
            type="dashed"
            block
            icon={<PlusOutlined />}
            onClick={addRow}
            className="mt-2"
          >
            Add Variable
          </Button>
        </div>

        <div className="flex items-center justify-between mt-6 pt-4 border-t border-gray-100">
          <Text type="secondary" className="text-xs">
            {savedAt
              ? `Last saved ${new Date(savedAt).toLocaleTimeString()} → ${
                  CREDENTIAL_STORE_LABELS[storeCfg.provider]
                }`
              : "Changes are not saved until you click Save."}
          </Text>
          <Button
            type="primary"
            icon={<SaveOutlined />}
            onClick={handleSave}
            loading={saving}
            disabled={!isDirty}
          >
            Save Variables
          </Button>
        </div>
      </Card>

      <CredentialStoreConfigModal
        open={storeModalOpen}
        cfg={storeCfg}
        onClose={() => setStoreModalOpen(false)}
        onSave={(c) => {
          saveCredentialStoreConfig(c);
          setStoreCfg(c);
          setStoreModalOpen(false);
        }}
      />
    </div>
  );
};

// ---------------------------------------------------------------------------

const KV: React.FC<{ k: string; v: string }> = ({ k, v }) => (
  <div className="flex items-baseline gap-2">
    <Text className="text-gray-500">{k}:</Text>
    <Text code className="break-all">
      {v}
    </Text>
  </div>
);

interface VariableRowProps {
  row: RowState;
  onNameChange: (v: string) => void;
  onValueChange: (v: string) => void;
  onToggleVisible: () => void;
  onRemove: () => void;
}

const VariableRow: React.FC<VariableRowProps> = ({
  row,
  onNameChange,
  onValueChange,
  onToggleVisible,
  onRemove,
}) => {
  // What the input renders:
  //   - if not stored yet (new row): plain editable input with toggle.
  //   - if stored but untouched: 8 dots placeholder, can't reveal.
  //   - if stored and modified this session: live value, can reveal.
  const isLockedDots = row.stored && !row.modifiedInSession;
  const displayValue = isLockedDots
    ? PLACEHOLDER_DOTS
    : row.visible
      ? row.value
      : row.value
        ? "•".repeat(Math.max(row.value.length, 4))
        : "";

  return (
    <div className="flex gap-3 items-center">
      <Form.Item className="mb-0" style={{ flex: 1 }}>
        <Input
          value={row.name}
          onChange={(e) => onNameChange(e.target.value.toUpperCase())}
          placeholder="e.g. CORP_PASSWORD"
          className="font-mono"
          disabled={row.stored && !row.modifiedInSession}
        />
      </Form.Item>
      <Form.Item className="mb-0" style={{ flex: 1.4 }}>
        <Input
          value={displayValue}
          onChange={(e) => onValueChange(e.target.value)}
          // When locked-dots we want the user's first keystroke to start
          // a fresh value (not append after the dots), so we clear on focus.
          onFocus={() => {
            if (isLockedDots) onValueChange("");
          }}
          placeholder={
            isLockedDots
              ? "Replace stored value…"
              : "Enter your value"
          }
          className="font-mono"
          suffix={
            <Tooltip
              title={
                row.modifiedInSession
                  ? row.visible
                    ? "Hide"
                    : "Show"
                  : "Replace the value before you can reveal it"
              }
            >
              <span
                className={`cursor-${row.modifiedInSession ? "pointer" : "not-allowed"}`}
                style={{
                  opacity: row.modifiedInSession ? 1 : 0.35,
                }}
                onClick={() => row.modifiedInSession && onToggleVisible()}
              >
                {row.visible ? <EyeOutlined /> : <EyeInvisibleOutlined />}
              </span>
            </Tooltip>
          }
        />
      </Form.Item>
      <div
        style={{ width: 130 }}
        className="flex items-center"
      >
        {row.stored && !row.modifiedInSession ? (
          <Tooltip
            title={
              row.updated_at
                ? `Last saved ${new Date(row.updated_at).toLocaleString()}`
                : undefined
            }
          >
            <Tag color="success" style={{ marginRight: 0 }}>
              Stored
            </Tag>
          </Tooltip>
        ) : row.modifiedInSession && row.stored ? (
          <Tag color="gold" style={{ marginRight: 0 }}>
            Will replace
          </Tag>
        ) : row.modifiedInSession ? (
          <Tag color="blue" style={{ marginRight: 0 }}>
            New
          </Tag>
        ) : (
          <Tag style={{ marginRight: 0 }}>—</Tag>
        )}
      </div>
      <div style={{ width: 32 }} className="flex items-center justify-center">
        <Popconfirm
          title="Delete this variable?"
          okText="Delete"
          okButtonProps={{ danger: true }}
          onConfirm={onRemove}
        >
          <Button size="small" type="text" danger>
            ✕
          </Button>
        </Popconfirm>
      </div>
    </div>
  );
};

// ---------------------------------------------------------------------------
// admin: choose credential store backend
// ---------------------------------------------------------------------------

interface CredentialStoreConfigModalProps {
  open: boolean;
  cfg: CredentialStoreConfig;
  onClose: () => void;
  onSave: (cfg: CredentialStoreConfig) => void;
}

const CredentialStoreConfigModal: React.FC<CredentialStoreConfigModalProps> = ({
  open,
  cfg,
  onClose,
  onSave,
}) => {
  const [provider, setProvider] = useState<CredentialStoreProvider>(
    cfg.provider,
  );
  const [hashicorp, setHashicorp] = useState(
    cfg.hashicorp ?? {
      address: "https://vault.example.com:8200",
      mount_path: "secret/litellm/users",
      namespace: "",
      auth_method: "token" as const,
    },
  );
  const [aws, setAws] = useState(
    cfg.aws ?? { region: "us-east-1", prefix: "litellm/users/" },
  );

  useEffect(() => {
    if (!open) return;
    setProvider(cfg.provider);
    if (cfg.hashicorp) setHashicorp(cfg.hashicorp);
    if (cfg.aws) setAws(cfg.aws);
  }, [open, cfg]);

  const handleSave = () => {
    const next: CredentialStoreConfig = { provider };
    if (provider === "hashicorp_vault") next.hashicorp = hashicorp;
    if (provider === "aws_secrets_manager") next.aws = aws;
    onSave(next);
  };

  return (
    <Modal
      open={open}
      onCancel={onClose}
      footer={null}
      title={
        <div className="flex items-center gap-2">
          <LockOutlined className="text-purple-500" />
          <span>Credential Store Backend</span>
          <Tag color="purple">Prototype</Tag>
        </div>
      }
      width={620}
    >
      <Alert
        type="warning"
        showIcon
        className="mb-4"
        message="UI-only override (prototype)"
        description={
          <span className="text-sm">
            In production this is read from <code>config.yaml</code> /
            environment variables, not editable in the UI. Shown here so the
            customer can preview the flow.
          </span>
        }
      />

      <Paragraph className="text-sm mb-1">
        <Text strong>Provider</Text>
      </Paragraph>
      <Select<CredentialStoreProvider>
        value={provider}
        onChange={(v) => setProvider(v)}
        style={{ width: "100%" }}
        size="large"
        options={(
          Object.keys(CREDENTIAL_STORE_LABELS) as CredentialStoreProvider[]
        ).map((p) => ({ value: p, label: CREDENTIAL_STORE_LABELS[p] }))}
      />

      {provider === "hashicorp_vault" && (
        <div className="mt-4 space-y-3">
          <LabelledInput
            label="Vault Address"
            value={hashicorp.address}
            onChange={(v) => setHashicorp({ ...hashicorp, address: v })}
            placeholder="https://vault.example.com:8200"
          />
          <LabelledInput
            label="Mount Path"
            value={hashicorp.mount_path}
            onChange={(v) => setHashicorp({ ...hashicorp, mount_path: v })}
            placeholder="secret/litellm/users"
          />
          <LabelledInput
            label="Namespace (optional)"
            value={hashicorp.namespace ?? ""}
            onChange={(v) => setHashicorp({ ...hashicorp, namespace: v })}
            placeholder="admin"
          />
          <div>
            <div className="text-sm font-medium mb-1">Auth Method</div>
            <Select
              value={hashicorp.auth_method}
              onChange={(v) =>
                setHashicorp({
                  ...hashicorp,
                  auth_method: v as "token" | "approle" | "kubernetes",
                })
              }
              style={{ width: "100%" }}
              options={[
                { value: "token", label: "Token (VAULT_TOKEN env var)" },
                { value: "approle", label: "AppRole" },
                { value: "kubernetes", label: "Kubernetes service account" },
              ]}
            />
          </div>

          <Alert
            type="info"
            showIcon
            className="mt-2"
            message={
              <span className="text-xs">
                Equivalent <code>config.yaml</code>:
              </span>
            }
            description={
              <pre className="text-xs bg-white border border-gray-200 rounded p-2 m-0 overflow-x-auto">
{`credential_store:
  provider: hashicorp_vault
  hashicorp:
    address: ${hashicorp.address}
    mount_path: ${hashicorp.mount_path}${
      hashicorp.namespace
        ? `
    namespace: ${hashicorp.namespace}`
        : ""
    }
    auth_method: ${hashicorp.auth_method}`}
              </pre>
            }
          />
        </div>
      )}

      {provider === "aws_secrets_manager" && (
        <div className="mt-4 space-y-3">
          <LabelledInput
            label="Region"
            value={aws.region}
            onChange={(v) => setAws({ ...aws, region: v })}
            placeholder="us-east-1"
          />
          <LabelledInput
            label="Secret Prefix"
            value={aws.prefix}
            onChange={(v) => setAws({ ...aws, prefix: v })}
            placeholder="litellm/users/"
          />
        </div>
      )}

      {(provider === "gcp_secret_manager" ||
        provider === "azure_key_vault" ||
        provider === "litellm_db") && (
        <div className="mt-4">
          <Alert
            type="info"
            showIcon
            message={
              provider === "litellm_db"
                ? "Default backend. Per-user variables are stored encrypted in the LiteLLM database."
                : "Configuration for this provider is read from environment variables in the prototype."
            }
          />
        </div>
      )}

      <div className="flex items-center justify-end gap-2 mt-6 pt-4 border-t border-gray-100">
        <Button onClick={onClose}>Cancel</Button>
        <Button type="primary" onClick={handleSave}>
          Save Configuration
        </Button>
      </div>
    </Modal>
  );
};

const LabelledInput: React.FC<{
  label: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
}> = ({ label, value, onChange, placeholder }) => (
  <div>
    <div className="text-sm font-medium mb-1">{label}</div>
    <Input
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      className="font-mono"
    />
  </div>
);

export default VariablesTab;
