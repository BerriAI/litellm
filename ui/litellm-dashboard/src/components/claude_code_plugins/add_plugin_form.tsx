import React, { useState } from "react";
import { Modal, Form, Input, Select } from "antd";
import { useTranslation } from "react-i18next";
import MessageManager from "@/components/molecules/message_manager";
import { Button } from "@tremor/react";
import { registerClaudeCodePlugin } from "../networking";
import { validatePluginName, isValidSemanticVersion, isValidEmail, isValidUrl, parseKeywords } from "./helpers";

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
  let s = raw
    .trim()
    .replace(/^https?:\/\//, "")
    .replace(/\/+$/, "");

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
  if (parts.length >= 5 && (parts[2] === "tree" || parts[2] === "blob")) {
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

const AddPluginForm: React.FC<AddPluginFormProps> = ({ visible, onClose, accessToken, onSuccess }) => {
  const { t } = useTranslation();
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
      MessageManager.error(t("claudeCodePluginsPage.addPluginForm.noAccessToken"));
      return;
    }

    if (!urlPreview) {
      MessageManager.error(t("claudeCodePluginsPage.addPluginForm.enterValidGithubUrl"));
      return;
    }

    if (!validatePluginName(values.name)) {
      MessageManager.error(t("claudeCodePluginsPage.addPluginForm.invalidSkillName"));
      return;
    }

    if (values.version && !isValidSemanticVersion(values.version)) {
      MessageManager.error(t("claudeCodePluginsPage.addPluginForm.invalidVersion"));
      return;
    }

    if (values.authorEmail && !isValidEmail(values.authorEmail)) {
      MessageManager.error(t("claudeCodePluginsPage.addPluginForm.invalidEmail"));
      return;
    }

    if (values.homepage && !isValidUrl(values.homepage)) {
      MessageManager.error(t("claudeCodePluginsPage.addPluginForm.invalidHomepageUrl"));
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
      MessageManager.success(t("claudeCodePluginsPage.addPluginForm.registerSuccess"));
      form.resetFields();
      setUrlPreview(null);
      onSuccess();
      onClose();
    } catch (error) {
      console.error("Error registering skill:", error);
      MessageManager.error(t("claudeCodePluginsPage.addPluginForm.registerFailed"));
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
      title={t("claudeCodePluginsPage.addPluginForm.modalTitle")}
      open={visible}
      onCancel={handleCancel}
      footer={null}
      width={700}
      className="top-8"
    >
      <Form form={form} layout="vertical" onFinish={handleSubmit} className="mt-4">
        {/* Smart URL Input */}
        <Form.Item
          label={t("claudeCodePluginsPage.addPluginForm.githubUrlLabel")}
          name="skillUrl"
          rules={[{ required: true, message: t("claudeCodePluginsPage.addPluginForm.githubUrlRequired") }]}
          tooltip={t("claudeCodePluginsPage.addPluginForm.githubUrlTooltip")}
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
            {t("claudeCodePluginsPage.addPluginForm.detected", { label: urlPreview.label })}
          </div>
        )}

        {/* Skill Name */}
        <Form.Item
          label={t("claudeCodePluginsPage.addPluginForm.skillNameLabel")}
          name="name"
          rules={[
            { required: true, message: t("claudeCodePluginsPage.addPluginForm.skillNameRequired") },
            {
              pattern: /^[a-z0-9-]+$/,
              message: t("claudeCodePluginsPage.addPluginForm.skillNamePattern"),
            },
          ]}
          tooltip={t("claudeCodePluginsPage.addPluginForm.skillNameTooltip")}
        >
          <Input placeholder="my-skill" className="rounded-lg" />
        </Form.Item>

        {/* Domain and Namespace — side by side */}
        <div className="flex gap-4">
          <Form.Item
            label={t("claudeCodePluginsPage.addPluginForm.domainLabel")}
            name="domain"
            tooltip={t("claudeCodePluginsPage.addPluginForm.domainTooltip")}
            className="flex-1"
          >
            <Input placeholder={t("claudeCodePluginsPage.addPluginForm.domainPlaceholder")} className="rounded-lg" />
          </Form.Item>
          <Form.Item
            label={t("claudeCodePluginsPage.addPluginForm.namespaceLabel")}
            name="namespace"
            tooltip={t("claudeCodePluginsPage.addPluginForm.namespaceTooltip")}
            className="flex-1"
          >
            <Input placeholder={t("claudeCodePluginsPage.addPluginForm.namespacePlaceholder")} className="rounded-lg" />
          </Form.Item>
        </div>

        {/* Description */}
        <Form.Item
          label={t("claudeCodePluginsPage.addPluginForm.descriptionLabel")}
          name="description"
          tooltip={t("claudeCodePluginsPage.addPluginForm.descriptionTooltip")}
        >
          <TextArea
            rows={3}
            placeholder={t("claudeCodePluginsPage.addPluginForm.descriptionPlaceholder")}
            maxLength={500}
            className="rounded-lg"
          />
        </Form.Item>

        {/* Category */}
        <Form.Item
          label={t("claudeCodePluginsPage.addPluginForm.categoryLabel")}
          name="category"
          tooltip={t("claudeCodePluginsPage.addPluginForm.categoryTooltip")}
        >
          <Select
            placeholder={t("claudeCodePluginsPage.addPluginForm.categoryPlaceholder")}
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
          label={t("claudeCodePluginsPage.addPluginForm.keywordsLabel")}
          name="keywords"
          tooltip={t("claudeCodePluginsPage.addPluginForm.keywordsTooltip")}
        >
          <Input placeholder={t("claudeCodePluginsPage.addPluginForm.keywordsPlaceholder")} className="rounded-lg" />
        </Form.Item>

        {/* Version */}
        <Form.Item
          label={t("claudeCodePluginsPage.addPluginForm.versionLabel")}
          name="version"
          tooltip={t("claudeCodePluginsPage.addPluginForm.versionTooltip")}
        >
          <Input placeholder="1.0.0" className="rounded-lg" />
        </Form.Item>

        {/* Author Name */}
        <Form.Item
          label={t("claudeCodePluginsPage.addPluginForm.authorNameLabel")}
          name="authorName"
          tooltip={t("claudeCodePluginsPage.addPluginForm.authorNameTooltip")}
        >
          <Input placeholder={t("claudeCodePluginsPage.addPluginForm.authorNamePlaceholder")} className="rounded-lg" />
        </Form.Item>

        {/* Author Email */}
        <Form.Item
          label={t("claudeCodePluginsPage.addPluginForm.authorEmailLabel")}
          name="authorEmail"
          rules={[{ type: "email", message: t("claudeCodePluginsPage.addPluginForm.authorEmailRule") }]}
          tooltip={t("claudeCodePluginsPage.addPluginForm.authorEmailTooltip")}
        >
          <Input type="email" placeholder="author@example.com" className="rounded-lg" />
        </Form.Item>

        {/* Submit Buttons */}
        <Form.Item className="mb-0 mt-6">
          <div className="flex justify-end gap-2">
            <Button variant="secondary" onClick={handleCancel} disabled={isSubmitting}>
              {t("common.cancel")}
            </Button>
            <Button type="submit" loading={isSubmitting}>
              {isSubmitting
                ? t("claudeCodePluginsPage.addPluginForm.adding")
                : t("claudeCodePluginsPage.addPluginForm.addSkill")}
            </Button>
          </div>
        </Form.Item>
      </Form>
    </Modal>
  );
};

export default AddPluginForm;
