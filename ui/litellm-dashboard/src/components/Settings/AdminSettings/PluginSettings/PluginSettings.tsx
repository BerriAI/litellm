"use client";

import { useState, useEffect } from "react";
import { Card, Space, Table, Typography } from "antd";
import { Button } from "@/components/ui/button";
import { DeleteOutlined, EditOutlined, PlusOutlined } from "@ant-design/icons";
import { Eye, EyeOff } from "lucide-react";
import { getConfigFieldSetting, updateConfigFieldSetting } from "@/components/networking";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { FieldGroup } from "@/components/shared/form/field";
import { FormField } from "@/components/shared/form/FormField";
import { Input } from "@/components/ui/input";
import { InputGroup, InputGroupAddon, InputGroupButton, InputGroupInput } from "@/components/ui/input-group";
import { useZodForm } from "@/lib/forms/useZodForm";
import { pluginSchema, type PluginFormValues } from "./schema";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";

const { Title, Text, Paragraph } = Typography;

interface Plugin {
  name: string;
  display_name: string;
  url: string;
  plugin_key?: string;
}

const BLANK_PLUGIN: PluginFormValues = { name: "", display_name: "", url: "", plugin_key: undefined };

export default function PluginSettings() {
  const { accessToken } = useAuthorized();
  const [plugins, setPlugins] = useState<Plugin[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [keyVisible, setKeyVisible] = useState(false);
  const form = useZodForm(pluginSchema, { defaultValues: BLANK_PLUGIN });

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
    setKeyVisible(false);
    form.reset(BLANK_PLUGIN);
    setModalOpen(true);
  };

  const openEdit = (idx: number) => {
    setEditingIndex(idx);
    setKeyVisible(false);
    // plugin_key arrives redacted ("***"); start it blank so an untouched save
    // keeps the stored credential instead of overwriting it with the placeholder.
    form.reset({ ...plugins[idx], plugin_key: "" });
    setModalOpen(true);
  };

  const handleDelete = (idx: number) => {
    const updated = plugins.filter((_, i) => i !== idx);
    save(updated);
  };

  const handleOk = async (values: PluginFormValues) => {
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
          <Button variant="outline" size="icon-sm" onClick={() => openEdit(idx)}>
            <EditOutlined />
          </Button>
          <Button variant="destructive" size="icon-sm" onClick={() => handleDelete(idx)}>
            <DeleteOutlined />
          </Button>
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

      <Button className="mb-4" onClick={openAdd}>
        <PlusOutlined />
        Add Plugin
      </Button>

      <Table dataSource={plugins} columns={columns} rowKey="name" loading={loading} pagination={false} size="small" />

      <Dialog open={modalOpen} onOpenChange={(open) => !open && setModalOpen(false)}>
        <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>{editingIndex !== null ? "Edit Plugin" : "Add Plugin"}</DialogTitle>
          </DialogHeader>
          <form onSubmit={(event) => event.preventDefault()} noValidate style={{ marginTop: 16 }}>
            <FieldGroup>
              <FormField
                control={form.control}
                name="name"
                label="Name (identifier)"
                description="Used in URLs and config. No spaces. E.g. litellm-platform-plugin"
              >
                {({ ref, ...field }) => <Input {...field} ref={ref} placeholder="litellm-platform-plugin" />}
              </FormField>
              <FormField control={form.control} name="display_name" label="Display Name">
                {({ ref, ...field }) => <Input {...field} ref={ref} placeholder="Agent Control Plane" />}
              </FormField>
              <FormField control={form.control} name="url" label="URL" description="Base URL of the plugin service">
                {({ ref, ...field }) => <Input {...field} ref={ref} placeholder="https://your-plugin.example.com" />}
              </FormField>
              <FormField
                control={form.control}
                name="plugin_key"
                label="Plugin Key"
                description="Optional. The plugin's own credential, injected as Authorization: Bearer <key> only when litellm reverse-proxies API calls to the plugin's backend (/plugin-proxy/<name>/*). Leave blank for plugins that use the forwarded litellm user token (e.g. iframe plugins) — that path uses the user's token, not this key."
              >
                {({ ref, ...field }) => (
                  <InputGroup>
                    <InputGroupInput
                      {...field}
                      ref={ref}
                      type={keyVisible ? "text" : "password"}
                      value={field.value ?? ""}
                      placeholder={editingIndex !== null ? "Leave blank to keep current key" : "sk-... (optional)"}
                    />
                    <InputGroupAddon align="inline-end">
                      <InputGroupButton
                        size="icon-xs"
                        onClick={() => setKeyVisible(!keyVisible)}
                        aria-label={keyVisible ? "Hide plugin key" : "Show plugin key"}
                      >
                        {keyVisible ? <EyeOff /> : <Eye />}
                      </InputGroupButton>
                    </InputGroupAddon>
                  </InputGroup>
                )}
              </FormField>
            </FieldGroup>
          </form>
          <DialogFooter>
            <Button variant="outline" onClick={() => setModalOpen(false)}>
              Cancel
            </Button>
            <Button onClick={form.handleSubmit(handleOk)} disabled={saving} aria-busy={saving}>
              Save
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Card>
  );
}
