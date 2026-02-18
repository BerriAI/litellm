import React, { useState } from "react";
import { Modal, Form, Input, Select, message } from "antd";
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

const AddPluginForm: React.FC<AddPluginFormProps> = ({
  visible,
  onClose,
  accessToken,
  onSuccess,
}) => {
  const [form] = Form.useForm();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [sourceType, setSourceType] = useState<"github" | "url">("github");

  const handleSubmit = async (values: any) => {
    if (!accessToken) {
      message.error("No access token available");
      return;
    }

    // Validate plugin name
    if (!validatePluginName(values.name)) {
      message.error(
        "Plugin name must be kebab-case (lowercase letters, numbers, and hyphens only)"
      );
      return;
    }

    // Validate semantic version if provided
    if (values.version && !isValidSemanticVersion(values.version)) {
      message.error(
        "Version must be in semantic versioning format (e.g., 1.0.0)"
      );
      return;
    }

    // Validate email if provided
    if (values.authorEmail && !isValidEmail(values.authorEmail)) {
      message.error("Invalid email format");
      return;
    }

    // Validate homepage URL if provided
    if (values.homepage && !isValidUrl(values.homepage)) {
      message.error("Invalid homepage URL format");
      return;
    }

    setIsSubmitting(true);
    try {
      // Build plugin data
      const pluginData: any = {
        name: values.name.trim(),
        source:
          sourceType === "github"
            ? {
                source: "github",
                repo: values.repo.trim(),
              }
            : {
                source: "url",
                url: values.url.trim(),
              },
      };

      // Add optional fields
      if (values.version) {
        pluginData.version = values.version.trim();
      }
      if (values.description) {
        pluginData.description = values.description.trim();
      }
      if (values.authorName || values.authorEmail) {
        pluginData.author = {};
        if (values.authorName) {
          pluginData.author.name = values.authorName.trim();
        }
        if (values.authorEmail) {
          pluginData.author.email = values.authorEmail.trim();
        }
      }
      if (values.homepage) {
        pluginData.homepage = values.homepage.trim();
      }
      if (values.category) {
        pluginData.category = values.category;
      }
      if (values.keywords) {
        pluginData.keywords = parseKeywords(values.keywords);
      }

      await registerClaudeCodePlugin(accessToken, pluginData);
      message.success("Plugin registered successfully");
      form.resetFields();
      setSourceType("github");
      onSuccess();
      onClose();
    } catch (error) {
      console.error("Error registering plugin:", error);
      message.error("Failed to register plugin");
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleCancel = () => {
    form.resetFields();
    setSourceType("github");
    onClose();
  };

  const handleSourceTypeChange = (value: "github" | "url") => {
    setSourceType(value);
    // Clear repo/url fields when switching
    form.setFieldsValue({ repo: undefined, url: undefined });
  };

  return (
    <Modal
      title="Add New Claude Code Plugin"
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
        {/* Plugin Name */}
        <Form.Item
          label="Plugin Name"
          name="name"
          rules={[
            { required: true, message: "Please enter plugin name" },
            {
              pattern: /^[a-z0-9-]+$/,
              message:
                "Name must be kebab-case (lowercase, numbers, hyphens only)",
            },
          ]}
          tooltip="Unique identifier in kebab-case format (e.g., my-awesome-plugin)"
        >
          <Input placeholder="my-awesome-plugin" className="rounded-lg" />
        </Form.Item>

        {/* Source Type */}
        <Form.Item
          label="Source Type"
          name="sourceType"
          initialValue="github"
          rules={[{ required: true, message: "Please select source type" }]}
        >
          <Select onChange={handleSourceTypeChange} className="rounded-lg">
            <Option value="github">GitHub</Option>
            <Option value="url">URL</Option>
          </Select>
        </Form.Item>

        {/* GitHub Repository */}
        {sourceType === "github" && (
          <Form.Item
            label="GitHub Repository"
            name="repo"
            rules={[
              { required: true, message: "Please enter repository" },
              {
                pattern: /^[a-zA-Z0-9_-]+\/[a-zA-Z0-9_-]+$/,
                message: "Repository must be in format: org/repo",
              },
            ]}
            tooltip="Format: organization/repository (e.g., anthropics/claude-code)"
          >
            <Input placeholder="anthropics/claude-code" className="rounded-lg" />
          </Form.Item>
        )}

        {/* Git URL */}
        {sourceType === "url" && (
          <Form.Item
            label="Git URL"
            name="url"
            rules={[{ required: true, message: "Please enter git URL" }]}
            tooltip="Full git URL to the repository"
          >
            <Input
              type="url"
              placeholder="https://github.com/org/repo.git"
              className="rounded-lg"
            />
          </Form.Item>
        )}

        {/* Version */}
        <Form.Item
          label="Version (Optional)"
          name="version"
          tooltip="Semantic version (e.g., 1.0.0)"
        >
          <Input placeholder="1.0.0" className="rounded-lg" />
        </Form.Item>

        {/* Description */}
        <Form.Item
          label="Description (Optional)"
          name="description"
          tooltip="Brief description of what the plugin does"
        >
          <TextArea
            rows={3}
            placeholder="A plugin that helps with..."
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

        {/* Author Name */}
        <Form.Item
          label="Author Name (Optional)"
          name="authorName"
          tooltip="Name of the plugin author or organization"
        >
          <Input placeholder="Your Name or Organization" className="rounded-lg" />
        </Form.Item>

        {/* Author Email */}
        <Form.Item
          label="Author Email (Optional)"
          name="authorEmail"
          rules={[{ type: "email", message: "Please enter a valid email" }]}
          tooltip="Contact email for the plugin author"
        >
          <Input type="email" placeholder="author@example.com" className="rounded-lg" />
        </Form.Item>

        {/* Homepage */}
        <Form.Item
          label="Homepage (Optional)"
          name="homepage"
          rules={[{ type: "url", message: "Please enter a valid URL" }]}
          tooltip="URL to the plugin's homepage or documentation"
        >
          <Input type="url" placeholder="https://example.com" className="rounded-lg" />
        </Form.Item>

        {/* Submit Buttons */}
        <Form.Item className="mb-0 mt-6">
          <div className="flex justify-end gap-2">
            <Button
              variant="secondary"
              onClick={handleCancel}
              disabled={isSubmitting}
            >
              Cancel
            </Button>
            <Button type="submit" loading={isSubmitting}>
              {isSubmitting ? "Registering..." : "Register Plugin"}
            </Button>
          </div>
        </Form.Item>
      </Form>
    </Modal>
  );
};

export default AddPluginForm;
