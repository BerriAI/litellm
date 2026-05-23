import { DeleteOutlined, PlusOutlined } from "@ant-design/icons";
import { Button, InputNumber, Space, Table, Typography } from "antd";
import { Input } from "antd";
import { useEffect, useState } from "react";

export interface TagRateLimitEntry {
  tag: string;
  rpm_limit: number | null;
  tpm_limit: number | null;
}

interface TagRateLimitEditorProps {
  /** Current value: { tag_rpm_limit: Record<string,number>, tag_tpm_limit: Record<string,number> } */
  value?: {
    tag_rpm_limit?: Record<string, number>;
    tag_tpm_limit?: Record<string, number>;
  };
  onChange?: (value: {
    tag_rpm_limit: Record<string, number>;
    tag_tpm_limit: Record<string, number>;
  }) => void;
  disabled?: boolean;
}

function entriesToMaps(entries: TagRateLimitEntry[]): {
  tag_rpm_limit: Record<string, number>;
  tag_tpm_limit: Record<string, number>;
} {
  const tag_rpm_limit: Record<string, number> = {};
  const tag_tpm_limit: Record<string, number> = {};
  for (const e of entries) {
    if (e.tag && e.rpm_limit != null) tag_rpm_limit[e.tag] = e.rpm_limit;
    if (e.tag && e.tpm_limit != null) tag_tpm_limit[e.tag] = e.tpm_limit;
  }
  return { tag_rpm_limit, tag_tpm_limit };
}

function mapsToEntries(
  tag_rpm_limit: Record<string, number> = {},
  tag_tpm_limit: Record<string, number> = {}
): TagRateLimitEntry[] {
  const tags = new Set([...Object.keys(tag_rpm_limit), ...Object.keys(tag_tpm_limit)]);
  return Array.from(tags).map((tag) => ({
    tag,
    rpm_limit: tag_rpm_limit[tag] ?? null,
    tpm_limit: tag_tpm_limit[tag] ?? null,
  }));
}

export function TagRateLimitEditor({
  value,
  onChange,
  disabled = false,
}: TagRateLimitEditorProps) {
  const [entries, setEntries] = useState<TagRateLimitEntry[]>(() =>
    mapsToEntries(value?.tag_rpm_limit, value?.tag_tpm_limit)
  );

  useEffect(() => {
    setEntries(mapsToEntries(value?.tag_rpm_limit, value?.tag_tpm_limit));
  }, [value]);

  const notify = (updated: TagRateLimitEntry[]) => {
    onChange?.(entriesToMaps(updated));
  };

  const addRow = () => {
    const updated = [...entries, { tag: "", rpm_limit: null, tpm_limit: null }];
    setEntries(updated);
  };

  const removeRow = (index: number) => {
    const updated = entries.filter((_, i) => i !== index);
    setEntries(updated);
    notify(updated);
  };

  const updateField = <K extends keyof TagRateLimitEntry>(
    index: number,
    field: K,
    val: TagRateLimitEntry[K]
  ) => {
    const updated = entries.map((e, i) => (i === index ? { ...e, [field]: val } : e));
    setEntries(updated);
    notify(updated);
  };

  const columns = [
    {
      title: "Tag",
      dataIndex: "tag",
      render: (_: string, __: TagRateLimitEntry, index: number) => (
        <Input
          disabled={disabled}
          value={entries[index].tag}
          placeholder="e.g. cell-1"
          onChange={(e) => updateField(index, "tag", e.target.value)}
          style={{ width: "100%" }}
        />
      ),
    },
    {
      title: "RPM Limit",
      dataIndex: "rpm_limit",
      render: (_: number | null, __: TagRateLimitEntry, index: number) => (
        <InputNumber
          disabled={disabled}
          min={0}
          value={entries[index].rpm_limit}
          placeholder="e.g. 100"
          onChange={(val) => updateField(index, "rpm_limit", val)}
          style={{ width: "100%" }}
        />
      ),
    },
    {
      title: "TPM Limit",
      dataIndex: "tpm_limit",
      render: (_: number | null, __: TagRateLimitEntry, index: number) => (
        <InputNumber
          disabled={disabled}
          min={0}
          value={entries[index].tpm_limit}
          placeholder="e.g. 10000"
          onChange={(val) => updateField(index, "tpm_limit", val)}
          style={{ width: "100%" }}
        />
      ),
    },
    {
      title: "",
      width: 48,
      render: (_: unknown, __: TagRateLimitEntry, index: number) => (
        <Button
          type="text"
          disabled={disabled}
          danger
          icon={<DeleteOutlined />}
          onClick={() => removeRow(index)}
        />
      ),
    },
  ];

  return (
    <Space direction="vertical" style={{ width: "100%" }}>
      <Typography.Text type="secondary" style={{ fontSize: 12 }}>
        Set independent RPM/TPM limits per request tag. Requests without a matching tag fall back to
        the key-level limits.
      </Typography.Text>
      <Table
        dataSource={entries}
        columns={columns}
        pagination={false}
        size="small"
        rowKey={(_, index) => String(index)}
      />
      <Button
        type="dashed"
        disabled={disabled}
        icon={<PlusOutlined />}
        onClick={addRow}
        style={{ width: "100%" }}
      >
        Add Tag Rate Limit
      </Button>
    </Space>
  );
}
