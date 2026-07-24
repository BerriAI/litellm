"use client";

import { useState, useEffect } from "react";
import { Button, Card, Form, Input, Modal, Space, Table, Typography } from "antd";
import { DeleteOutlined, EditOutlined, PlusOutlined } from "@ant-design/icons";
import { getConfigFieldSetting, updateConfigFieldSetting } from "@/components/networking";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

const { Title, Text, Paragraph } = Typography;

interface Plugin {
  name: string;
  display_name: string;
  url: string;
  plugin_key?: string;
}

export default function PluginSettings() {
  const { accessToken } = useAuthorized();
  const [plugins, setPlugins] = useState<Plugin[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [form] = Form.useForm<Plugin>();

  useEffect(() => {
    if (!accessToken) return;
    getConfigFieldSetting(accessToken, "plugins")
      .then((data) => {
        const val = data?.field_value;
        setPlugins(Array.isArray(val) ? val : []);
      })
      .catch(() => setPlugins([]))
      .finally(() => setLoading(false));
  }, [accessToken]);

  const save = async (updated: Plugin[]) => {
    if (!accessToken) return;
    setSaving(true);
    try {
      await updateConfigFieldSetting(accessToken, "plugins", updated);
      setPlugins(updated);
    } finally {
      setSaving(false);
    }
  };

  const openAdd = () => {
    setEditingIndex(null);
    form.resetFields();
    setModalOpen(true);
  };

  const openEdit = (idx: number) => {
    setEditingIndex(idx);
    // plugin_key arrives redacted ("***"); start it blank so an untouched save
    // keeps the stored credential instead of overwriting it with the placeholder.
    form.setFieldsValue({ ...plugins[idx], plugin_key: "" });
    setModalOpen(true);
  };

  const handleDelete = (idx: number) => {
    const updated = plugins.filter((_, i) => i !== idx);
    save(updated);
  };

  const handleOk = async () => {
    const values = await form.validateFields();
    const updated =
      editingIndex !== null ? plugins.map((p, i) => (i === editingIndex ? values : p)) : [...plugins, values];
    await save(updated);
    setModalOpen(false);
  };

  const columns = [
    {
      title: "Name",
      dataIndex: "name",
      key: "name",
      render: (v: string) => <Text code>{v}</Text>,
    },
    { title: "Display Name", dataIndex: "display_name", key: "display_name" },
    {
      title: "URL",
      dataIndex: "url",
      key: "url",
      render: (v: string) => (
        <a href={v} target="_blank" rel="noopener noreferrer">
          {v}
        </a>
      ),
    },
    {
      title: "Plugin Key",
      dataIndex: "plugin_key",
      key: "plugin_key",
      render: (v?: string) => (v ? <Text code>{"•".repeat(8)}</Text> : <Text type="secondary">—</Text>),
    },
    {
      title: "Actions",
      key: "actions",
      render: (_: unknown, __: Plugin, idx: number) => (
        <Space>
          <Button icon={<EditOutlined />} size="small" onClick={() => openEdit(idx)} />
          <Button icon={<DeleteOutlined />} size="small" danger onClick={() => handleDelete(idx)} />
        </Space>
      ),
    },
  ];

  return (
    <Card>
      <Title level={4}>Plugins</Title>
      <Paragraph>
        Register external services as plugins. Once added, users can toggle to the plugin from the mode switcher in the
        top-left of the sidebar.
      </Paragraph>
      <Paragraph type="secondary" style={{ fontSize: 12 }}>
        Each plugin must expose <Text code>GET /api/plugin-manifest</Text> returning nav items and capabilities.
      </Paragraph>

      <Button type="primary" icon={<PlusOutlined />} onClick={openAdd} style={{ marginBottom: 16 }}>
        Add Plugin
      </Button>

      <Table dataSource={plugins} columns={columns} rowKey="name" loading={loading} pagination={false} size="small" />

      <Modal
        title={editingIndex !== null ? "Edit Plugin" : "Add Plugin"}
        open={modalOpen}
        onOk={handleOk}
        onCancel={() => setModalOpen(false)}
        confirmLoading={saving}
        okText="Save"
      >
        <Form form={form} layout="vertical" style={{ marginTop: 16 }}>
          <Form.Item
            name="name"
            label="Name (identifier)"
            rules={[{ required: true, message: "Required" }]}
            extra="Used in URLs and config. No spaces. E.g. litellm-platform-plugin"
          >
            <Input placeholder="litellm-platform-plugin" />
          </Form.Item>
          <Form.Item name="display_name" label="Display Name" rules={[{ required: true, message: "Required" }]}>
            <Input placeholder="Agent Control Plane" />
          </Form.Item>
          <Form.Item
            name="url"
            label="URL"
            rules={[
              { required: true, message: "Required" },
              { type: "url", message: "Must be a valid URL" },
            ]}
            extra="Base URL of the plugin service"
          >
            <Input placeholder="https://your-plugin.example.com" />
          </Form.Item>
          <Form.Item
            name="plugin_key"
            label="Plugin Key"
            extra="Optional. The plugin's own credential, injected as Authorization: Bearer <key> only when litellm reverse-proxies API calls to the plugin's backend (/plugin-proxy/<name>/*). Leave blank for plugins that use the forwarded litellm user token (e.g. iframe plugins) — that path uses the user's token, not this key."
          >
            <Input.Password
              placeholder={editingIndex !== null ? "Leave blank to keep current key" : "sk-... (optional)"}
            />
          </Form.Item>
        </Form>
      </Modal>
    </Card>
  );
}
