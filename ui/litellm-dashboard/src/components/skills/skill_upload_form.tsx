/**
 * Skill upload form (S2-09).
 *
 * Two tabs:
 *  - ZIP — drop in a manifest.yaml + SKILL.md bundle, hits the upload endpoint.
 *  - Form — fill display_title + system_prompt_template + tool_schema by hand.
 *
 * Both paths land in the same backend (POST /v1/xct-skills/upload or /v1/xct-skills).
 */

import React, { useState } from "react";
import {
  Alert,
  Button,
  Checkbox,
  Form,
  Input,
  Space,
  Tabs,
  Upload,
  Typography,
} from "antd";
import type { UploadFile } from "antd/es/upload/interface";
import { InboxOutlined } from "@ant-design/icons";
import { createXCTSkill, uploadXCTSkillZip } from "../networking";

const { Text } = Typography;
const { Dragger } = Upload;

interface Props {
  accessToken: string;
  onSuccess: (skill: any) => void;
}

const SkillUploadForm: React.FC<Props> = ({ accessToken, onSuccess }) => {
  const [submitting, setSubmitting] = useState(false);
  const [zipFile, setZipFile] = useState<UploadFile | null>(null);
  const [isPublicOverride, setIsPublicOverride] = useState<boolean | undefined>(undefined);

  // ---- ZIP submit -------------------------------------------------------
  const handleZipSubmit = async () => {
    if (!zipFile?.originFileObj) return;
    setSubmitting(true);
    try {
      const created = await uploadXCTSkillZip(
        accessToken,
        zipFile.originFileObj as File,
        isPublicOverride !== undefined ? { is_public_override: isPublicOverride } : {},
      );
      onSuccess(created);
    } finally {
      setSubmitting(false);
    }
  };

  // ---- Manual form submit ----------------------------------------------
  const [form] = Form.useForm();
  const handleFormSubmit = async (values: any) => {
    setSubmitting(true);
    try {
      // tool_schema is entered as JSON text; parse before sending.
      let toolSchema: any = undefined;
      if (values.tool_schema && values.tool_schema.trim()) {
        try {
          toolSchema = JSON.parse(values.tool_schema);
        } catch {
          form.setFields([{ name: "tool_schema", errors: ["Must be valid JSON"] }]);
          setSubmitting(false);
          return;
        }
      }
      const created = await createXCTSkill(accessToken, {
        display_title: values.display_title,
        description: values.description,
        version: values.version || "1",
        is_public: !!values.is_public,
        system_prompt_template: values.system_prompt_template,
        tool_schema: toolSchema,
      });
      onSuccess(created);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Tabs
      defaultActiveKey="zip"
      items={[
        {
          key: "zip",
          label: "Upload ZIP",
          children: (
            <Space direction="vertical" size="middle" style={{ width: "100%" }}>
              <Alert
                type="info"
                showIcon
                message={
                  <span>
                    Required: <code>manifest.yaml</code> at the archive root.
                    Optional: <code>SKILL.md</code>, <code>tools.json</code>,{" "}
                    <code>README.md</code>. Max 10 MB.
                  </span>
                }
              />
              <Dragger
                accept=".zip,application/zip"
                multiple={false}
                maxCount={1}
                beforeUpload={(file) => {
                  setZipFile({ uid: file.uid, name: file.name, originFileObj: file } as UploadFile);
                  return false; // don't auto-upload — we POST on Submit
                }}
                onRemove={() => setZipFile(null)}
              >
                <p className="ant-upload-drag-icon">
                  <InboxOutlined />
                </p>
                <p className="ant-upload-text">Click or drag a skill ZIP here</p>
                <p className="ant-upload-hint">
                  manifest.yaml must include a <code>display_title</code> (or{" "}
                  <code>name</code>).
                </p>
              </Dragger>
              <Checkbox
                checked={isPublicOverride === true}
                onChange={(e) =>
                  setIsPublicOverride(e.target.checked ? true : undefined)
                }
              >
                Override manifest's <code>is_public</code> to <strong>true</strong>
              </Checkbox>
              <Button
                type="primary"
                disabled={!zipFile}
                loading={submitting}
                onClick={handleZipSubmit}
              >
                Upload
              </Button>
            </Space>
          ),
        },
        {
          key: "form",
          label: "Manual",
          children: (
            <Form
              form={form}
              layout="vertical"
              onFinish={handleFormSubmit}
              initialValues={{ version: "1" }}
            >
              <Form.Item
                name="display_title"
                label="Display title"
                rules={[{ required: true, message: "Required" }]}
              >
                <Input placeholder="Fact Check" />
              </Form.Item>
              <Form.Item name="description" label="Description">
                <Input.TextArea rows={2} placeholder="Short summary shown in the marketplace" />
              </Form.Item>
              <Form.Item name="version" label="Version">
                <Input placeholder="1" />
              </Form.Item>
              <Form.Item name="is_public" valuePropName="checked">
                <Checkbox>Public (visible to anonymous /.well-known/xct-capabilities)</Checkbox>
              </Form.Item>
              <Form.Item
                name="system_prompt_template"
                label="system_prompt_template"
              >
                <Input.TextArea
                  rows={8}
                  placeholder={"You are a fact-checking assistant. Cite every claim with...\nSupports Jinja2 {{ variable }} placeholders."}
                />
              </Form.Item>
              <Form.Item
                name="tool_schema"
                label="tool_schema (JSON array of OpenAI-shape tools)"
              >
                <Input.TextArea
                  rows={6}
                  placeholder='[\n  {"type":"function","function":{"name":"cite","description":"...","parameters":{}}}\n]'
                />
              </Form.Item>
              <Space>
                <Button type="primary" htmlType="submit" loading={submitting}>
                  Create skill
                </Button>
                <Text type="secondary">
                  After creating, click <strong>Publish</strong> in the list to freeze
                  the content fields.
                </Text>
              </Space>
            </Form>
          ),
        },
      ]}
    />
  );
};

export default SkillUploadForm;
