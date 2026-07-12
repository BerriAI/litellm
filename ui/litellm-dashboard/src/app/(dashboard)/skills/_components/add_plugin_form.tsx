import React, { useState } from "react";
import { Modal, Form, Input, Select } from "antd";
import MessageManager from "@/components/molecules/message_manager";
import { Button } from "@tremor/react";
import { registerClaudeCodePlugin } from "@/components/networking";
import {
  validatePluginName,
  isValidSemanticVersion,
  isValidEmail,
  isValidUrl,
  parseKeywords,
  parseSkillSource,
  isValidSubPath,
  SkillSourcePreview,
} from "@/components/claude_code_plugins/helpers";
import { PluginAuthor, PluginSource, SkillRegisterRequest } from "@/components/claude_code_plugins/types";

const { TextArea } = Input;
const { Option } = Select;

interface AddPluginFormProps {
  visible: boolean;
  onClose: () => void;
  accessToken: string | null;
  onSuccess: () => void;
}

interface AddPluginFormValues {
  name: string;
  skillUrl?: string;
  subPath?: string;
  version?: string;
  description?: string;
  authorName?: string;
  authorEmail?: string;
  homepage?: string;
  category?: string;
  keywords?: string;
  domain?: string;
  namespace?: string;
}

const buildAuthor = (values: AddPluginFormValues): PluginAuthor | undefined => {
  const name = values.authorName?.trim();
  const email = values.authorEmail?.trim();
  if (!name) {
    return undefined;
  }
  return email ? { name, email } : { name };
};

const buildRegisterRequest = (values: AddPluginFormValues, source: PluginSource): SkillRegisterRequest => {
  const author = buildAuthor(values);
  return {
    name: values.name.trim(),
    source,
    ...(values.version ? { version: values.version.trim() } : {}),
    ...(values.description ? { description: values.description.trim() } : {}),
    ...(author ? { author } : {}),
    ...(values.homepage ? { homepage: values.homepage.trim() } : {}),
    ...(values.category ? { category: values.category } : {}),
    ...(values.keywords ? { keywords: parseKeywords(values.keywords) } : {}),
    ...(values.domain ? { domain: values.domain.trim() } : {}),
    ...(values.namespace ? { namespace: values.namespace.trim() } : {}),
  };
};

const PREDEFINED_CATEGORIES = [
  "Development",
  "Productivity",
  "Learning",
  "Security",
  "Data & Analytics",
  "Integration",
  "Testing",
  "Documentation",
];

const AddPluginForm: React.FC<AddPluginFormProps> = ({ visible, onClose, accessToken, onSuccess }) => {
  const [form] = Form.useForm();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [urlPreview, setUrlPreview] = useState<SkillSourcePreview | null>(null);
  const [urlEncodesSubdir, setUrlEncodesSubdir] = useState(false);

  const recomputePreview = (skillUrl: string, subPath: string) => {
    const encodesSubdir = parseSkillSource(skillUrl)?.parsed.source === "git-subdir";
    setUrlEncodesSubdir(encodesSubdir);
    if (encodesSubdir && form.getFieldValue("subPath")) {
      form.setFieldsValue({ subPath: "" });
    }
    const preview = parseSkillSource(skillUrl, encodesSubdir ? undefined : subPath);
    setUrlPreview(preview);
    if (preview && !form.getFieldValue("name")) {
      form.setFieldsValue({ name: preview.suggestedName });
    }
  };

  const handleUrlChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    recomputePreview(e.target.value, form.getFieldValue("subPath") ?? "");
  };

  const handleSubPathChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    recomputePreview(form.getFieldValue("skillUrl") ?? "", e.target.value);
  };

  const handleSubmit = async (values: AddPluginFormValues) => {
    if (!accessToken) {
      MessageManager.error("No access token available");
      return;
    }

    if (!urlPreview) {
      MessageManager.error("Please enter a valid repository URL");
      return;
    }

    if (!validatePluginName(values.name)) {
      MessageManager.error("Skill name must be kebab-case (lowercase letters, numbers, and hyphens only)");
      return;
    }

    if (values.version && !isValidSemanticVersion(values.version)) {
      MessageManager.error("Version must be in semantic versioning format (e.g., 1.0.0)");
      return;
    }

    if (values.authorEmail && !isValidEmail(values.authorEmail)) {
      MessageManager.error("Invalid email format");
      return;
    }

    if (values.homepage && !isValidUrl(values.homepage)) {
      MessageManager.error("Invalid homepage URL format");
      return;
    }

    setIsSubmitting(true);
    try {
      await registerClaudeCodePlugin(accessToken, buildRegisterRequest(values, urlPreview.parsed));
      MessageManager.success("Skill registered successfully");
      form.resetFields();
      setUrlPreview(null);
      setUrlEncodesSubdir(false);
      onSuccess();
      onClose();
    } catch (error) {
      console.error("Error registering skill:", error);
      const reason = error instanceof Error && error.message ? error.message : "Failed to register skill";
      MessageManager.error(`Failed to register skill: ${reason}`);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleCancel = () => {
    form.resetFields();
    setUrlPreview(null);
    setUrlEncodesSubdir(false);
    onClose();
  };

  return (
    <Modal title="Add New Skill" open={visible} onCancel={handleCancel} footer={null} width={700} className="top-8">
      <Form form={form} layout="vertical" onFinish={handleSubmit} className="mt-4">
        {/* Smart URL Input */}
        <Form.Item
          label="Repository URL"
          name="skillUrl"
          rules={[{ required: true, message: "Please enter a repository URL" }]}
          tooltip="Paste an HTTPS git repository URL from GitHub, GitLab, Bitbucket, or a self-hosted host. E.g. github.com/org/repo, gitlab.com/org/repo, or github.com/org/repo/tree/main/my-skill"
        >
          <Input
            placeholder="https://github.com/org/repo or https://gitlab.com/org/repo"
            className="rounded-lg"
            onChange={handleUrlChange}
          />
        </Form.Item>

        {/* Optional subfolder for monorepos */}
        <Form.Item
          label="Subfolder path (Optional)"
          name="subPath"
          rules={[
            {
              validator: (_, value) =>
                !value || isValidSubPath(value)
                  ? Promise.resolve()
                  : Promise.reject(
                      new Error(
                        "Subfolder must be a relative path like plugins/my-skill (letters, numbers, dots, hyphens, underscores)",
                      ),
                    ),
            },
          ]}
          tooltip="Path within the repository where the skill lives (e.g., plugins/my-skill). Leave empty if the skill is at the repo root."
          extra={urlEncodesSubdir ? "The URL already points to a subfolder, so this field is disabled" : undefined}
        >
          <Input
            placeholder="plugins/my-skill"
            className="rounded-lg"
            onChange={handleSubPathChange}
            disabled={urlEncodesSubdir}
          />
        </Form.Item>

        {/* Parsed preview */}
        {urlPreview && (
          <div className="mb-4 px-3 py-2 bg-blue-50 border border-blue-200 rounded-lg text-sm text-blue-700">
            Detected: {urlPreview.label}
          </div>
        )}

        {/* Skill Name */}
        <Form.Item
          label="Skill Name"
          name="name"
          rules={[
            { required: true, message: "Please enter skill name" },
            {
              pattern: /^[a-z0-9-]+$/,
              message: "Name must be kebab-case (lowercase, numbers, hyphens only)",
            },
          ]}
          tooltip="Unique identifier in kebab-case format (e.g., my-skill)"
        >
          <Input placeholder="my-skill" className="rounded-lg" />
        </Form.Item>

        {/* Domain and Namespace — side by side */}
        <div className="flex gap-4">
          <Form.Item
            label="Domain (Optional)"
            name="domain"
            tooltip="Top-level grouping in the Skill Hub (e.g., Productivity)"
            className="flex-1"
          >
            <Input placeholder="Productivity" className="rounded-lg" />
          </Form.Item>
          <Form.Item
            label="Namespace (Optional)"
            name="namespace"
            tooltip="Sub-grouping within domain (e.g., workflows)"
            className="flex-1"
          >
            <Input placeholder="workflows" className="rounded-lg" />
          </Form.Item>
        </div>

        {/* Description */}
        <Form.Item label="Description (Optional)" name="description" tooltip="Brief description of what the skill does">
          <TextArea rows={3} placeholder="A skill that helps with..." maxLength={500} className="rounded-lg" />
        </Form.Item>

        {/* Category */}
        <Form.Item label="Category (Optional)" name="category" tooltip="Select a category or enter a custom one">
          <Select
            placeholder="Select or type a category"
            allowClear
            showSearch
            optionFilterProp="children"
            className="rounded-lg"
          >
            {PREDEFINED_CATEGORIES.map((cat) => (
              <Option key={cat} value={cat}>
                {cat}
              </Option>
            ))}
          </Select>
        </Form.Item>

        {/* Keywords */}
        <Form.Item label="Keywords (Optional)" name="keywords" tooltip="Comma-separated list of keywords for search">
          <Input placeholder="search, web, api" className="rounded-lg" />
        </Form.Item>

        {/* Version */}
        <Form.Item label="Version (Optional)" name="version" tooltip="Semantic version (e.g., 1.0.0)">
          <Input placeholder="1.0.0" className="rounded-lg" />
        </Form.Item>

        {/* Author Name */}
        <Form.Item label="Author Name (Optional)" name="authorName" tooltip="Name of the skill author or organization">
          <Input placeholder="Your Name or Organization" className="rounded-lg" />
        </Form.Item>

        {/* Author Email */}
        <Form.Item
          label="Author Email (Optional)"
          name="authorEmail"
          rules={[{ type: "email", message: "Please enter a valid email" }]}
          tooltip="Contact email for the skill author"
        >
          <Input type="email" placeholder="author@example.com" className="rounded-lg" />
        </Form.Item>

        {/* Submit Buttons */}
        <Form.Item className="mb-0 mt-6">
          <div className="flex justify-end gap-2">
            <Button variant="secondary" onClick={handleCancel} disabled={isSubmitting}>
              Cancel
            </Button>
            <Button type="submit" loading={isSubmitting}>
              {isSubmitting ? "Adding..." : "Add Skill"}
            </Button>
          </div>
        </Form.Item>
      </Form>
    </Modal>
  );
};

export default AddPluginForm;
