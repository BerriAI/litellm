import { useProxyConnection, ProxyConnection } from "@/contexts/ProxyConnectionContext";
import {
  CheckCircleFilled,
  CloudServerOutlined,
  DeleteOutlined,
  EditOutlined,
  PlusOutlined,
} from "@ant-design/icons";
import { Button, Form, Input, List, Modal, Space, Tag, Typography, message } from "antd";
import React, { useState } from "react";

const { Text } = Typography;

interface ManageProxiesModalProps {
  open: boolean;
  onClose: () => void;
}

interface ConnectionForm {
  name: string;
  url: string;
  apiKey: string;
}

const ManageProxiesModal: React.FC<ManageProxiesModalProps> = ({ open, onClose }) => {
  const { connections, activeConnection, addConnection, updateConnection, removeConnection, testConnection } =
    useProxyConnection();

  const [form] = Form.useForm<ConnectionForm>();
  const [editingId, setEditingId] = useState<string | null>(null);
  const [showForm, setShowForm] = useState(false);
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<{ ok: boolean; message: string } | null>(null);

  const resetForm = () => {
    form.resetFields();
    setEditingId(null);
    setShowForm(false);
    setTestResult(null);
  };

  const handleTest = async () => {
    try {
      const values = await form.validateFields(["url", "apiKey"]);
      setTesting(true);
      setTestResult(null);
      const result = await testConnection(values.url, values.apiKey);
      if (result.ok) {
        setTestResult({ ok: true, message: `Connected successfully (v${result.version || "unknown"})` });
      } else {
        setTestResult({ ok: false, message: result.error || "Connection failed" });
      }
    } catch {
      // form validation error
    } finally {
      setTesting(false);
    }
  };

  const handleSave = async () => {
    try {
      const values = await form.validateFields();
      if (editingId) {
        updateConnection(editingId, values);
        message.success("Connection updated");
      } else {
        addConnection(values);
        message.success("Connection added");
      }
      resetForm();
    } catch {
      // form validation error
    }
  };

  const handleEdit = (conn: ProxyConnection) => {
    setEditingId(conn.id);
    setShowForm(true);
    setTestResult(null);
    form.setFieldsValue({
      name: conn.name,
      url: conn.url,
      apiKey: conn.apiKey,
    });
  };

  const handleDelete = (conn: ProxyConnection) => {
    Modal.confirm({
      title: "Remove Connection",
      content: `Are you sure you want to remove "${conn.name}"?`,
      okText: "Remove",
      okType: "danger",
      onOk: () => {
        removeConnection(conn.id);
        message.success("Connection removed");
      },
    });
  };

  return (
    <Modal
      title={
        <Space>
          <CloudServerOutlined />
          Manage Proxy Connections
        </Space>
      }
      open={open}
      onCancel={() => {
        resetForm();
        onClose();
      }}
      footer={null}
      width={600}
    >
      <List
        dataSource={connections}
        renderItem={(conn) => (
          <List.Item
            actions={
              conn.isDefault
                ? [
                    <Tag key="default" color={conn.id === activeConnection?.id ? "green" : "default"}>
                      {conn.id === activeConnection?.id ? "Active" : "Local"}
                    </Tag>,
                  ]
                : [
                    conn.id === activeConnection?.id && (
                      <Tag key="active" color="green">
                        Active
                      </Tag>
                    ),
                    <Button key="edit" type="text" size="small" icon={<EditOutlined />} onClick={() => handleEdit(conn)} />,
                    <Button
                      key="delete"
                      type="text"
                      size="small"
                      danger
                      icon={<DeleteOutlined />}
                      onClick={() => handleDelete(conn)}
                    />,
                  ].filter(Boolean)
            }
          >
            <List.Item.Meta
              title={
                <Space>
                  {conn.id === activeConnection?.id && <CheckCircleFilled style={{ color: "#52c41a" }} />}
                  {conn.name}
                </Space>
              }
              description={<Text type="secondary" ellipsis>{conn.url}</Text>}
            />
          </List.Item>
        )}
      />

      {showForm ? (
        <div style={{ marginTop: 16, padding: 16, border: "1px solid #f0f0f0", borderRadius: 8 }}>
          <Text strong style={{ marginBottom: 12, display: "block" }}>
            {editingId ? "Edit Connection" : "Add Connection"}
          </Text>
          <Form form={form} layout="vertical" size="small">
            <Form.Item name="name" label="Name" rules={[{ required: true, message: "Enter a name for this connection" }]}>
              <Input placeholder="e.g. Production US-East" />
            </Form.Item>
            <Form.Item
              name="url"
              label="Proxy URL"
              rules={[
                { required: true, message: "Enter the proxy URL" },
                { type: "url", message: "Enter a valid URL" },
              ]}
            >
              <Input placeholder="e.g. https://litellm-prod.example.com" />
            </Form.Item>
            <Form.Item
              name="apiKey"
              label="API Key"
              rules={[{ required: true, message: "Enter an API key for this proxy" }]}
              tooltip="A master key or virtual key for authenticating with the remote proxy"
            >
              <Input.Password placeholder="sk-..." />
            </Form.Item>

            {testResult && (
              <div
                style={{
                  marginBottom: 12,
                  padding: 8,
                  borderRadius: 4,
                  backgroundColor: testResult.ok ? "#f6ffed" : "#fff2f0",
                  border: `1px solid ${testResult.ok ? "#b7eb8f" : "#ffccc7"}`,
                }}
              >
                <Text type={testResult.ok ? "success" : "danger"}>{testResult.message}</Text>
              </div>
            )}

            <Space>
              <Button onClick={handleTest} loading={testing}>
                Test Connection
              </Button>
              <Button type="primary" onClick={handleSave}>
                {editingId ? "Update" : "Add"}
              </Button>
              <Button onClick={resetForm}>Cancel</Button>
            </Space>
          </Form>
        </div>
      ) : (
        <Button
          type="dashed"
          icon={<PlusOutlined />}
          onClick={() => {
            setShowForm(true);
            setEditingId(null);
            setTestResult(null);
          }}
          style={{ marginTop: 16, width: "100%" }}
        >
          Add Connection
        </Button>
      )}
    </Modal>
  );
};

export default ManageProxiesModal;
