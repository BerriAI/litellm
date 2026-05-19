/**
 * XCT Apps — list / create / rotate / delete (S4-09).
 *
 * After create + rotate-secret the cleartext is shown ONCE in a non-dismissable
 * "copy to clipboard" modal. We deliberately do not put the secret in
 * sessionStorage — once the modal closes it's unrecoverable.
 */

import React, { useEffect, useState } from "react";
import {
  Alert,
  Button,
  Card,
  Form,
  Input,
  InputNumber,
  Modal,
  Space,
  Switch,
  Table,
  Tag,
  Tooltip,
  Typography,
  message,
} from "antd";
import {
  CopyOutlined,
  DeleteOutlined,
  ReloadOutlined,
  RetweetOutlined,
} from "@ant-design/icons";
import {
  createXCTApp,
  deleteXCTApp,
  listXCTApps,
  patchXCTApp,
  rotateXCTAppSecret,
} from "../networking";

const { Text, Title } = Typography;

interface AppRow {
  app_id: string;
  app_name: string;
  display_name: string;
  description?: string;
  oauth_client_id: string;
  redirect_uris: string[];
  default_team_id?: string;
  default_scopes: string[];
  capability_scope_id?: string;
  rpm_limit?: number;
  daily_budget?: number;
  is_active: boolean;
  created_at?: string;
}

interface Props {
  accessToken: string;
}

const SecretRevealModal: React.FC<{
  open: boolean;
  secret?: string;
  clientId?: string;
  onClose: () => void;
}> = ({ open, secret, clientId, onClose }) => (
  <Modal
    title="Save this secret now — it will not be shown again"
    open={open}
    onCancel={onClose}
    onOk={onClose}
    okText="I've saved it"
    cancelButtonProps={{ style: { display: "none" } }}
    closable={false}
    maskClosable={false}
    width={580}
  >
    <Alert
      type="warning"
      showIcon
      message="The proxy stores only a hash. If you lose this value, rotate the secret to generate a new one."
      style={{ marginBottom: 16 }}
    />
    <Space direction="vertical" size="small" style={{ width: "100%" }}>
      {clientId ? (
        <>
          <Text strong>oauth_client_id</Text>
          <Input.Group compact>
            <Input value={clientId} readOnly style={{ width: "calc(100% - 40px)" }} />
            <Button
              icon={<CopyOutlined />}
              onClick={() => {
                navigator.clipboard?.writeText(clientId);
                message.success("Copied");
              }}
            />
          </Input.Group>
        </>
      ) : null}
      <Text strong>client_secret</Text>
      <Input.Group compact>
        <Input.Password
          value={secret}
          readOnly
          visibilityToggle
          style={{ width: "calc(100% - 40px)" }}
        />
        <Button
          icon={<CopyOutlined />}
          onClick={() => {
            if (secret) navigator.clipboard?.writeText(secret);
            message.success("Copied");
          }}
        />
      </Input.Group>
    </Space>
  </Modal>
);

const AppTable: React.FC<Props> = ({ accessToken }) => {
  const [rows, setRows] = useState<AppRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [createOpen, setCreateOpen] = useState(false);
  const [secretReveal, setSecretReveal] = useState<{
    secret?: string;
    clientId?: string;
  } | null>(null);

  const refresh = async () => {
    setLoading(true);
    try {
      const data = await listXCTApps(accessToken);
      setRows(Array.isArray(data) ? data : []);
    } catch (e: any) {
      message.error(`Load failed: ${e.message ?? e}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (accessToken) refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [accessToken]);

  const handleRotate = (row: AppRow) => {
    Modal.confirm({
      title: `Rotate secret for ${row.display_name}?`,
      content:
        "The old secret will be invalid immediately. Update the app's deployment before any in-flight token exchange.",
      okText: "Rotate",
      onOk: async () => {
        try {
          const res = await rotateXCTAppSecret(accessToken, row.app_id);
          setSecretReveal({
            secret: res.client_secret,
            clientId: res.oauth_client_id,
          });
          refresh();
        } catch (e: any) {
          message.error(`Rotate failed: ${e.message ?? e}`);
        }
      },
    });
  };

  const handleToggleActive = async (row: AppRow) => {
    try {
      await patchXCTApp(accessToken, row.app_id, { is_active: !row.is_active });
      refresh();
    } catch (e: any) {
      message.error(`Update failed: ${e.message ?? e}`);
    }
  };

  const handleDelete = (row: AppRow) => {
    Modal.confirm({
      title: `Delete ${row.display_name}?`,
      content:
        "The app's OAuth credentials become invalid. Existing access tokens remain in the DB for audit but cannot be refreshed.",
      okText: "Delete",
      okType: "danger",
      onOk: async () => {
        await deleteXCTApp(accessToken, row.app_id);
        refresh();
      },
    });
  };

  const columns = [
    {
      title: "App",
      key: "name",
      render: (_: any, row: AppRow) => (
        <Space direction="vertical" size={0}>
          <Text strong>{row.display_name}</Text>
          <Text type="secondary" style={{ fontSize: 12 }}>
            {row.app_name}
          </Text>
        </Space>
      ),
    },
    {
      title: "client_id",
      dataIndex: "oauth_client_id",
      key: "oauth_client_id",
      render: (v: string) => (
        <Space>
          <Text code>{v}</Text>
          <Button
            type="text"
            icon={<CopyOutlined />}
            size="small"
            onClick={() => {
              navigator.clipboard?.writeText(v);
              message.success("Copied");
            }}
          />
        </Space>
      ),
    },
    {
      title: "Redirect URIs",
      dataIndex: "redirect_uris",
      key: "redirect_uris",
      render: (uris: string[]) =>
        uris && uris.length ? (
          <Space direction="vertical" size={0}>
            {uris.map((u) => (
              <Text key={u} type="secondary" style={{ fontSize: 12 }}>
                {u}
              </Text>
            ))}
          </Space>
        ) : (
          <Text type="secondary">—</Text>
        ),
    },
    {
      title: "Scope",
      dataIndex: "capability_scope_id",
      key: "capability_scope_id",
      render: (v?: string) => (v ? <Tag color="blue">{v}</Tag> : <Tag>—</Tag>),
    },
    {
      title: "Active",
      dataIndex: "is_active",
      key: "is_active",
      width: 90,
      render: (v: boolean, row: AppRow) => (
        <Switch checked={v} onChange={() => handleToggleActive(row)} />
      ),
    },
    {
      title: "Actions",
      key: "actions",
      width: 160,
      render: (_: any, row: AppRow) => (
        <Space>
          <Tooltip title="Rotate secret">
            <Button
              type="text"
              icon={<RetweetOutlined />}
              onClick={() => handleRotate(row)}
            />
          </Tooltip>
          <Tooltip title="Delete">
            <Button
              type="text"
              danger
              icon={<DeleteOutlined />}
              onClick={() => handleDelete(row)}
            />
          </Tooltip>
        </Space>
      ),
    },
  ];

  const [createForm] = Form.useForm();
  const submitCreate = async (values: any) => {
    try {
      const payload = {
        app_name: values.app_name,
        display_name: values.display_name,
        description: values.description,
        redirect_uris: (values.redirect_uris || "")
          .split("\n")
          .map((s: string) => s.trim())
          .filter(Boolean),
        default_team_id: values.default_team_id || undefined,
        default_scopes: (values.default_scopes || "")
          .split(/[\s,]+/)
          .map((s: string) => s.trim())
          .filter(Boolean),
        capability_scope_id: values.capability_scope_id || undefined,
        rpm_limit: values.rpm_limit ?? undefined,
        daily_budget: values.daily_budget ?? undefined,
      };
      const res = await createXCTApp(accessToken, payload);
      setCreateOpen(false);
      createForm.resetFields();
      setSecretReveal({ secret: res.client_secret, clientId: res.oauth_client_id });
      refresh();
    } catch (e: any) {
      message.error(`Create failed: ${e.message ?? e}`);
    }
  };

  return (
    <Card
      title={<Title level={4}>XCT Apps</Title>}
      extra={
        <Space>
          <Button icon={<ReloadOutlined />} onClick={refresh}>
            Refresh
          </Button>
          <Button type="primary" onClick={() => setCreateOpen(true)}>
            New app
          </Button>
        </Space>
      }
    >
      <Table
        rowKey="app_id"
        loading={loading}
        columns={columns as any}
        dataSource={rows}
        pagination={{ pageSize: 20, showSizeChanger: false }}
      />

      <Modal
        title="Create XCT App"
        open={createOpen}
        onCancel={() => setCreateOpen(false)}
        onOk={() => createForm.submit()}
        okText="Create"
        width={620}
      >
        <Form form={createForm} layout="vertical" onFinish={submitCreate}>
          <Form.Item
            name="app_name"
            label="app_name (slug, unique)"
            rules={[{ required: true, message: "Required" }]}
          >
            <Input placeholder="xct-chat" />
          </Form.Item>
          <Form.Item
            name="display_name"
            label="Display name"
            rules={[{ required: true, message: "Required" }]}
          >
            <Input placeholder="XCT Chat" />
          </Form.Item>
          <Form.Item name="description" label="Description">
            <Input.TextArea rows={2} />
          </Form.Item>
          <Form.Item
            name="redirect_uris"
            label="Redirect URIs (one per line, exact-match enforced)"
          >
            <Input.TextArea
              rows={3}
              placeholder="https://chat.xct.test/oauth/callback&#10;https://chat.xct.io/oauth/callback"
            />
          </Form.Item>
          <Form.Item name="default_team_id" label="Default team_id">
            <Input placeholder="t-XYZ (tokens issued via OAuth attach to this team)" />
          </Form.Item>
          <Form.Item name="default_scopes" label="Default scopes (space- or comma-separated)">
            <Input placeholder="read write" />
          </Form.Item>
          <Form.Item name="capability_scope_id" label="capability_scope_id (access group)">
            <Input placeholder="grp-XYZ — narrows /v1/capabilities to this group's allow lists" />
          </Form.Item>
          <Space>
            <Form.Item name="rpm_limit" label="RPM limit">
              <InputNumber min={0} />
            </Form.Item>
            <Form.Item name="daily_budget" label="Daily budget (USD)">
              <InputNumber min={0} step={1} />
            </Form.Item>
          </Space>
        </Form>
      </Modal>

      <SecretRevealModal
        open={secretReveal !== null}
        secret={secretReveal?.secret}
        clientId={secretReveal?.clientId}
        onClose={() => setSecretReveal(null)}
      />
    </Card>
  );
};

export default AppTable;
