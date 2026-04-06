import React, { useState } from "react";
import { Modal, Form, Input, Select } from "antd";
import MessageManager from "@/components/molecules/message_manager";
import { Button } from "@tremor/react";
import { registerClaudeCodePlugin } from "../networking";
import {
  validatePluginName,
  isValidSemanticVersion,
  isValidEmail,
  isValidUrl,
  parseKeywords,
} from "./helpers";

const { TextArea } = Input;
const { Option } = Select;

interface AddPluginFormProps {
  visible: boolean;
  onClose: () => void;
  accessToken: string | null;
  onSuccess: () => void;
}

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

interface ParsedSource {
  source: "github" | "url" | "git-subdir";
  repo?: string;
  url?: string;
  path?: string;
}

interface ParsePreview {
  parsed: ParsedSource;
  label: string;
  suggestedName: string;
}

function parseGitHubUrl(raw: string): ParsePreview | null {
  // Strip protocol and trailing slashes/spaces
  let s = raw.trim().replace(/^https?:\/\//, "").replace(/\/+$/, "");

  if (!s.startsWith("github.com/")) return null;

  // Remove "github.com/"
  const rest = s.slice("github.com/".length);
  const parts = rest.split("/");

  if (parts.length < 2) return null;

  const org = parts[0];
  const repo = parts[1];
  const repoBase = repo.replace(/\.git$/, "");

  // github.com/org/repo  (exactly 2 parts, or ends with .git)
  if (parts.length === 2 || (parts.length === 2 && repoBase)) {
    return {
      parsed: { source: "github", repo: `${org}/${repoBase}` },
      label: `GitHub repo — ${org}/${repoBase}`,
      suggestedName: repoBase,
    };
  }

  // github.com/org/repo/tree/branch/folder or /blob/branch/folder/FILE.md
  if (
    parts.length >= 5 &&
    (parts[2] === "tree" || parts[2] === "blob")
  ) {
    // parts[3] = branch, parts[4..] = path segments
    const pathParts = parts.slice(4);
    // If last segment looks like a file (has extension), drop it
    const lastPart = pathParts[pathParts.length - 1];
    if (lastPart && lastPart.includes(".")) {
      pathParts.pop();
    }
    if (pathParts.length === 0) {
      // Path resolved to repo root — treat as plain github source
      return {
        parsed: { source: "github", repo: `${org}/${repoBase}` },
        label: `GitHub repo — ${org}/${repoBase}`,
        suggestedName: repoBase,
      };
    }
    const subPath = pathParts.join("/");
    const suggestedName = pathParts[pathParts.length - 1];
    return {
      parsed: {
        source: "git-subdir",
        url: `https://github.com/${org}/${repoBase}`,
        path: subPath,
      },
      label: `GitHub subdir — ${org}/${repoBase} @ ${subPath}`,
      suggestedName,
    };
  }

  return null;
}

const AddPluginForm: React.FC<AddPluginFormProps> = ({
  visible,
  onClose,
  accessToken,
  onSuccess,
}) => {
  const [form] = Form.useForm();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [urlPreview, setUrlPreview] = useState<ParsePreview | null>(null);

  const handleUrlChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    const preview = parseGitHubUrl(val);
    setUrlPreview(preview);
    if (preview) {
      // Auto-fill name only if it's currently empty
      const currentName = form.getFieldValue("name");
      if (!currentName) {
        form.setFieldsValue({ name: preview.suggestedName });
      }
    }
  };

  const handleSubmit = async (values: any) => {
    if (!accessToken) {
      MessageManager.error("No access token available");
      return;
    }

    if (!urlPreview) {
      MessageManager.error("Please enter a valid GitHub URL");
      return;
    }

    if (!validatePluginName(values.name)) {
      MessageManager.error(
        "Skill name must be kebab-case (lowercase letters, numbers, and hyphens only)"
      );
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
      const pluginData: any = {
        name: values.name.trim(),
        source: urlPreview.parsed,
      };

      if (values.version) pluginData.version = values.version.trim();
      if (values.description) pluginData.description = values.description.trim();
      if (values.authorName || values.authorEmail) {
        pluginData.author = {};
        if (values.authorName) pluginData.author.name = values.authorName.trim();
        if (values.authorEmail) pluginData.author.email = values.authorEmail.trim();
      }
      if (values.homepage) pluginData.homepage = values.homepage.trim();
      if (values.category) pluginData.category = values.category;
      if (values.keywords) pluginData.keywords = parseKeywords(values.keywords);
      if (values.domain) pluginData.domain = values.domain.trim();
      if (values.namespace) pluginData.namespace = values.namespace.trim();

      await registerClaudeCodePlugin(accessToken, pluginData);
      MessageManager.success("Skill registered successfully");
      form.resetFields();
      setUrlPreview(null);
      onSuccess();
      onClose();
    } catch (error) {
      console.error("Error registering skill:", error);
      MessageManager.error("Failed to register skill");
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleCancel = () => {
    form.resetFields();
    setUrlPreview(null);
    onClose();
  };

  return (
    <Modal
      title="Add New Skill"
      open={visible}
      onCancel={handleCancel}
      footer={null}
      width={700}
      className="top-8"
    >
      <Form
        form={form}
        layout="vertical"
        onFinish={handleSubmit}
        className="mt-4"
      >
        {/* Smart URL Input */}
        <Form.Item
          label="GitHub URL"
          name="skillUrl"
          rules={[{ required: true, message: "Please enter a GitHub URL" }]}
          tooltip="Paste a GitHub URL — repo, folder, or file link. E.g. github.com/org/repo or github.com/org/repo/tree/main/my-skill"
        >
          <Input
            placeholder="https://github.com/org/repo/tree/main/my-skill"
            className="rounded-lg"
            onChange={handleUrlChange}
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
        <Form.Item
          label="Description (Optional)"
          name="description"
          tooltip="Brief description of what the skill does"
        >
          <TextArea
            rows={3}
            placeholder="A skill that helps with..."
            maxLength={500}
            className="rounded-lg"
          />
        </Form.Item>

        {/* Category */}
        <Form.Item
          label="Category (Optional)"
          name="category"
          tooltip="Select a category or enter a custom one"
        >
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
        <Form.Item
          label="Keywords (Optional)"
          name="keywords"
          tooltip="Comma-separated list of keywords for search"
        >
          <Input placeholder="search, web, api" className="rounded-lg" />
        </Form.Item>

        {/* Version */}
        <Form.Item
          label="Version (Optional)"
          name="version"
          tooltip="Semantic version (e.g., 1.0.0)"
        >
          <Input placeholder="1.0.0" className="rounded-lg" />
        </Form.Item>

        {/* Author Name */}
        <Form.Item
          label="Author Name (Optional)"
          name="authorName"
          tooltip="Name of the skill author or organization"
        >
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
