import React, { useState } from "react";
import { useForm, Controller } from "react-hook-form";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import MessageManager from "@/components/molecules/message_manager";
import { registerClaudeCodePlugin } from "../networking";
import {
  validatePluginName,
  isValidSemanticVersion,
  isValidEmail,
  isValidUrl,
  parseKeywords,
} from "./helpers";

interface AddPluginFormProps {
  visible: boolean;
  onClose: () => void;
  accessToken: string | null;
  onSuccess: () => void;
}

interface FormValues {
  skillUrl: string;
  name: string;
  domain?: string;
  namespace?: string;
  description?: string;
  category?: string;
  keywords?: string;
  version?: string;
  authorName?: string;
  authorEmail?: string;
  homepage?: string;
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
  const form = useForm<FormValues>({
    defaultValues: {
      skillUrl: "",
      name: "",
      domain: "",
      namespace: "",
      description: "",
      category: "",
      keywords: "",
      version: "",
      authorName: "",
      authorEmail: "",
    },
  });
  const { register, handleSubmit, setValue, getValues, reset, control, formState } = form;
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [urlPreview, setUrlPreview] = useState<ParsePreview | null>(null);

  const handleUrlChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    const preview = parseGitHubUrl(val);
    setUrlPreview(preview);
    if (preview) {
      const currentName = getValues("name");
      if (!currentName) {
        setValue("name", preview.suggestedName, { shouldValidate: false });
      }
    }
  };

  const onSubmit = async (values: FormValues) => {
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
      reset();
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
    reset();
    setUrlPreview(null);
    onClose();
  };

  const urlRegister = register("skillUrl", {
    required: "Please enter a GitHub URL",
  });

  return (
    <Dialog
      open={visible}
      onOpenChange={(o) => (!o ? handleCancel() : undefined)}
    >
      <DialogContent className="max-w-[700px] max-h-[90vh] overflow-y-auto top-8">
        <DialogHeader>
          <DialogTitle>Add New Skill</DialogTitle>
        </DialogHeader>
        <form
          onSubmit={handleSubmit(onSubmit)}
          className="mt-4 space-y-4"
        >
          {/* Smart URL Input */}
          <div className="space-y-2">
            <Label htmlFor="skillUrl">
              GitHub URL <span className="text-destructive">*</span>
            </Label>
            <Input
              id="skillUrl"
              placeholder="https://github.com/org/repo/tree/main/my-skill"
              className="rounded-lg"
              aria-invalid={!!formState.errors.skillUrl}
              {...urlRegister}
              onChange={(e) => {
                urlRegister.onChange(e);
                handleUrlChange(e);
              }}
            />
            <p className="text-xs text-muted-foreground">
              Paste a GitHub URL — repo, folder, or file link. E.g.
              github.com/org/repo or github.com/org/repo/tree/main/my-skill
            </p>
            {formState.errors.skillUrl && (
              <p className="text-sm text-destructive">
                {formState.errors.skillUrl.message as string}
              </p>
            )}
          </div>

          {/* Parsed preview */}
          {urlPreview && (
            <div className="px-3 py-2 bg-blue-50 dark:bg-blue-950/30 border border-blue-200 dark:border-blue-900 rounded-lg text-sm text-blue-700 dark:text-blue-300">
              Detected: {urlPreview.label}
            </div>
          )}

          {/* Skill Name */}
          <div className="space-y-2">
            <Label htmlFor="name">
              Skill Name <span className="text-destructive">*</span>
            </Label>
            <Input
              id="name"
              placeholder="my-skill"
              className="rounded-lg"
              aria-invalid={!!formState.errors.name}
              {...register("name", {
                required: "Please enter skill name",
                pattern: {
                  value: /^[a-z0-9-]+$/,
                  message:
                    "Name must be kebab-case (lowercase, numbers, hyphens only)",
                },
              })}
            />
            <p className="text-xs text-muted-foreground">
              Unique identifier in kebab-case format (e.g., my-skill)
            </p>
            {formState.errors.name && (
              <p className="text-sm text-destructive">
                {formState.errors.name.message as string}
              </p>
            )}
          </div>

          {/* Domain and Namespace — side by side */}
          <div className="flex gap-4">
            <div className="flex-1 space-y-2">
              <Label htmlFor="domain">Domain (Optional)</Label>
              <Input
                id="domain"
                placeholder="Productivity"
                className="rounded-lg"
                {...register("domain")}
              />
              <p className="text-xs text-muted-foreground">
                Top-level grouping in the Skill Hub (e.g., Productivity)
              </p>
            </div>
            <div className="flex-1 space-y-2">
              <Label htmlFor="namespace">Namespace (Optional)</Label>
              <Input
                id="namespace"
                placeholder="workflows"
                className="rounded-lg"
                {...register("namespace")}
              />
              <p className="text-xs text-muted-foreground">
                Sub-grouping within domain (e.g., workflows)
              </p>
            </div>
          </div>

          {/* Description */}
          <div className="space-y-2">
            <Label htmlFor="description">Description (Optional)</Label>
            <Textarea
              id="description"
              rows={3}
              placeholder="A skill that helps with..."
              maxLength={500}
              className="rounded-lg"
              {...register("description")}
            />
            <p className="text-xs text-muted-foreground">
              Brief description of what the skill does
            </p>
          </div>

          {/* Category */}
          <div className="space-y-2">
            <Label htmlFor="category">Category (Optional)</Label>
            <Controller
              control={control}
              name="category"
              render={({ field }) => (
                <Select
                  value={field.value || ""}
                  onValueChange={(v) => field.onChange(v === "__none__" ? "" : v)}
                >
                  <SelectTrigger id="category" className="rounded-lg">
                    <SelectValue placeholder="Select a category" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="__none__">
                      <span className="text-muted-foreground">None</span>
                    </SelectItem>
                    {PREDEFINED_CATEGORIES.map((cat) => (
                      <SelectItem key={cat} value={cat}>
                        {cat}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              )}
            />
          </div>

          {/* Keywords */}
          <div className="space-y-2">
            <Label htmlFor="keywords">Keywords (Optional)</Label>
            <Input
              id="keywords"
              placeholder="search, web, api"
              className="rounded-lg"
              {...register("keywords")}
            />
            <p className="text-xs text-muted-foreground">
              Comma-separated list of keywords for search
            </p>
          </div>

          {/* Version */}
          <div className="space-y-2">
            <Label htmlFor="version">Version (Optional)</Label>
            <Input
              id="version"
              placeholder="1.0.0"
              className="rounded-lg"
              {...register("version")}
            />
            <p className="text-xs text-muted-foreground">
              Semantic version (e.g., 1.0.0)
            </p>
          </div>

          {/* Author Name */}
          <div className="space-y-2">
            <Label htmlFor="authorName">Author Name (Optional)</Label>
            <Input
              id="authorName"
              placeholder="Your Name or Organization"
              className="rounded-lg"
              {...register("authorName")}
            />
            <p className="text-xs text-muted-foreground">
              Name of the skill author or organization
            </p>
          </div>

          {/* Author Email */}
          <div className="space-y-2">
            <Label htmlFor="authorEmail">Author Email (Optional)</Label>
            <Input
              id="authorEmail"
              type="email"
              placeholder="author@example.com"
              className="rounded-lg"
              aria-invalid={!!formState.errors.authorEmail}
              {...register("authorEmail", {
                validate: (v) =>
                  !v || isValidEmail(v) || "Please enter a valid email",
              })}
            />
            <p className="text-xs text-muted-foreground">
              Contact email for the skill author
            </p>
            {formState.errors.authorEmail && (
              <p className="text-sm text-destructive">
                {formState.errors.authorEmail.message as string}
              </p>
            )}
          </div>

          {/* Submit Buttons */}
          <div className="flex justify-end gap-2 pt-6 mt-6">
            <Button
              type="button"
              variant="outline"
              onClick={handleCancel}
              disabled={isSubmitting}
            >
              Cancel
            </Button>
            <Button type="submit" disabled={isSubmitting}>
              {isSubmitting ? "Adding..." : "Add Skill"}
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
};

export default AddPluginForm;
