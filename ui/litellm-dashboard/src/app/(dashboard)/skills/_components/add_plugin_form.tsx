import React, { useState } from "react";
import { CircleHelp } from "lucide-react";
import { z } from "zod/v4";
import { toast } from "@/lib/toast";
import { registerClaudeCodePlugin } from "@/components/networking";
import { FieldGroup } from "@/components/ui/field";
import { FormField } from "@/components/shared/form/FormField";
import { Button } from "@/components/ui/button";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { useZodForm } from "@/lib/forms/useZodForm";
import {
  validatePluginName,
  isValidSemanticVersion,
  isValidEmail,
  parseKeywords,
  parseSkillSource,
  isValidSubPath,
  SkillSourcePreview,
} from "@/components/claude_code_plugins/helpers";
import { PluginAuthor, PluginSource, SkillRegisterRequest } from "@/components/claude_code_plugins/types";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";

interface AddPluginFormProps {
  visible: boolean;
  onClose: () => void;
  accessToken: string | null;
  onSuccess: () => void;
}

const addPluginShape = {
  skillUrl: z.string().min(1, "Please enter a repository URL"),
  subPath: z
    .string()
    .refine(
      (value) => !value || isValidSubPath(value),
      "Subfolder must be a relative path like plugins/my-skill (letters, numbers, dots, hyphens, underscores)",
    ),
  name: z
    .string()
    .min(1, "Please enter skill name")
    .regex(/^[a-z0-9-]+$/, "Name must be kebab-case (lowercase, numbers, hyphens only)"),
  domain: z.string(),
  namespace: z.string(),
  description: z.string(),
  category: z.string(),
  keywords: z.string(),
  version: z.string(),
  authorName: z.string(),
  authorEmail: z
    .string()
    .refine((value) => value === "" || z.email().safeParse(value).success, "Please enter a valid email"),
};

const addPluginSchema = z.object(addPluginShape);

type AddPluginFormValues = z.infer<typeof addPluginSchema>;

const EMPTY_VALUES: AddPluginFormValues = {
  skillUrl: "",
  subPath: "",
  name: "",
  domain: "",
  namespace: "",
  description: "",
  category: "",
  keywords: "",
  version: "",
  authorName: "",
  authorEmail: "",
};

const buildAuthor = (values: AddPluginFormValues): PluginAuthor | undefined => {
  const name = values.authorName.trim();
  const email = values.authorEmail.trim();
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

const labelWithHint = (label: string, hint: string): React.ReactNode => (
  <>
    {label}
    <Tooltip>
      <TooltipTrigger render={<CircleHelp className="size-3.5 shrink-0 cursor-help text-muted-foreground" />} />
      <TooltipContent>{hint}</TooltipContent>
    </Tooltip>
  </>
);

const AddPluginForm: React.FC<AddPluginFormProps> = ({ visible, onClose, accessToken, onSuccess }) => {
  const form = useZodForm(addPluginSchema, { defaultValues: EMPTY_VALUES });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [urlPreview, setUrlPreview] = useState<SkillSourcePreview | null>(null);
  const [urlEncodesSubdir, setUrlEncodesSubdir] = useState(false);

  const recomputePreview = (skillUrl: string, subPath: string) => {
    const encodesSubdir = parseSkillSource(skillUrl)?.parsed.source === "git-subdir";
    setUrlEncodesSubdir(encodesSubdir);
    if (encodesSubdir && form.getValues("subPath")) {
      form.setValue("subPath", "");
    }
    const preview = parseSkillSource(skillUrl, encodesSubdir ? undefined : subPath);
    setUrlPreview(preview);
    if (preview && !form.getValues("name")) {
      form.setValue("name", preview.suggestedName);
    }
  };

  const handleSubmit = async (values: AddPluginFormValues) => {
    if (!accessToken) {
      toast.error("No access token available");
      return;
    }

    if (!urlPreview) {
      toast.error("Please enter a valid repository URL");
      return;
    }

    if (!validatePluginName(values.name)) {
      toast.error("Skill name must be kebab-case (lowercase letters, numbers, and hyphens only)");
      return;
    }

    if (values.version && !isValidSemanticVersion(values.version)) {
      toast.error("Version must be in semantic versioning format (e.g., 1.0.0)");
      return;
    }

    if (values.authorEmail && !isValidEmail(values.authorEmail)) {
      toast.error("Invalid email format");
      return;
    }

    setIsSubmitting(true);
    try {
      await registerClaudeCodePlugin(accessToken, buildRegisterRequest(values, urlPreview.parsed));
      toast.success("Skill registered successfully");
      form.reset(EMPTY_VALUES);
      setUrlPreview(null);
      setUrlEncodesSubdir(false);
      onSuccess();
      onClose();
    } catch (error) {
      console.error("Error registering skill:", error);
      toast.error(error instanceof Error && error.message ? error.message : "Failed to register skill");
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleCancel = () => {
    form.reset(EMPTY_VALUES);
    setUrlPreview(null);
    setUrlEncodesSubdir(false);
    onClose();
  };

  return (
    <Dialog open={visible} onOpenChange={(open) => !open && handleCancel()}>
      <DialogContent className="top-8 max-h-[calc(100dvh-4rem)] translate-y-0 overflow-y-auto sm:max-w-[700px]">
        <DialogHeader>
          <DialogTitle>Add New Skill</DialogTitle>
        </DialogHeader>
        <TooltipProvider>
          <form onSubmit={form.handleSubmit(handleSubmit)} noValidate className="mt-4">
            <FieldGroup>
              <FormField
                control={form.control}
                name="skillUrl"
                label={labelWithHint(
                  "Repository URL",
                  "Paste an HTTPS git repository URL from GitHub, GitLab, Bitbucket, or a self-hosted host. E.g. github.com/org/repo, gitlab.com/org/repo, or github.com/org/repo/tree/main/my-skill",
                )}
              >
                {({ ref, onChange, ...field }) => (
                  <Input
                    {...field}
                    ref={ref}
                    placeholder="https://github.com/org/repo or https://gitlab.com/org/repo"
                    className="rounded-lg"
                    onChange={(event) => {
                      onChange(event);
                      recomputePreview(event.target.value, form.getValues("subPath"));
                    }}
                  />
                )}
              </FormField>

              <FormField
                control={form.control}
                name="subPath"
                label={labelWithHint(
                  "Subfolder path (Optional)",
                  "Path within the repository where the skill lives (e.g., plugins/my-skill). Leave empty if the skill is at the repo root.",
                )}
                description={
                  urlEncodesSubdir ? "The URL already points to a subfolder, so this field is disabled" : undefined
                }
              >
                {({ ref, onChange, ...field }) => (
                  <Input
                    {...field}
                    ref={ref}
                    placeholder="plugins/my-skill"
                    className="rounded-lg"
                    onChange={(event) => {
                      onChange(event);
                      recomputePreview(form.getValues("skillUrl"), event.target.value);
                    }}
                    disabled={urlEncodesSubdir}
                  />
                )}
              </FormField>

              {urlPreview && (
                <div className="rounded-lg border border-info/20 bg-info/10 px-3 py-2 text-sm text-info">
                  Detected: {urlPreview.label}
                </div>
              )}

              <FormField
                control={form.control}
                name="name"
                label={labelWithHint("Skill Name", "Unique identifier in kebab-case format (e.g., my-skill)")}
              >
                {({ ref, ...field }) => <Input {...field} ref={ref} placeholder="my-skill" className="rounded-lg" />}
              </FormField>

              <div className="flex gap-4">
                <FormField
                  control={form.control}
                  name="domain"
                  label={labelWithHint("Domain (Optional)", "Top-level grouping in the Skill Hub (e.g., Productivity)")}
                  className="flex-1"
                >
                  {({ ref, ...field }) => (
                    <Input {...field} ref={ref} placeholder="Productivity" className="rounded-lg" />
                  )}
                </FormField>
                <FormField
                  control={form.control}
                  name="namespace"
                  label={labelWithHint("Namespace (Optional)", "Sub-grouping within domain (e.g., workflows)")}
                  className="flex-1"
                >
                  {({ ref, ...field }) => <Input {...field} ref={ref} placeholder="workflows" className="rounded-lg" />}
                </FormField>
              </div>

              <FormField
                control={form.control}
                name="description"
                label={labelWithHint("Description (Optional)", "Brief description of what the skill does")}
              >
                {({ ref, ...field }) => (
                  <Textarea
                    {...field}
                    ref={ref}
                    rows={3}
                    placeholder="A skill that helps with..."
                    maxLength={500}
                    className="rounded-lg"
                  />
                )}
              </FormField>

              <FormField
                control={form.control}
                name="category"
                label={labelWithHint("Category (Optional)", "Select a category or enter a custom one")}
              >
                {({ id, value, onChange, "aria-invalid": ariaInvalid, "aria-describedby": ariaDescribedBy }) => (
                  <Combobox
                    items={PREDEFINED_CATEGORIES}
                    value={value === "" ? null : value}
                    onValueChange={(category: string | null) => onChange(category ?? "")}
                  >
                    <ComboboxInput
                      id={id}
                      aria-invalid={ariaInvalid}
                      aria-describedby={ariaDescribedBy}
                      placeholder="Select or type a category"
                      className="w-full rounded-lg"
                      showClear={value !== ""}
                    />
                    <ComboboxContent>
                      <ComboboxEmpty>No matching categories</ComboboxEmpty>
                      <ComboboxList>
                        {(category: string) => (
                          <ComboboxItem key={category} value={category}>
                            {category}
                          </ComboboxItem>
                        )}
                      </ComboboxList>
                    </ComboboxContent>
                  </Combobox>
                )}
              </FormField>

              <FormField
                control={form.control}
                name="keywords"
                label={labelWithHint("Keywords (Optional)", "Comma-separated list of keywords for search")}
              >
                {({ ref, ...field }) => (
                  <Input {...field} ref={ref} placeholder="search, web, api" className="rounded-lg" />
                )}
              </FormField>

              <FormField
                control={form.control}
                name="version"
                label={labelWithHint("Version (Optional)", "Semantic version (e.g., 1.0.0)")}
              >
                {({ ref, ...field }) => <Input {...field} ref={ref} placeholder="1.0.0" className="rounded-lg" />}
              </FormField>

              <FormField
                control={form.control}
                name="authorName"
                label={labelWithHint("Author Name (Optional)", "Name of the skill author or organization")}
              >
                {({ ref, ...field }) => (
                  <Input {...field} ref={ref} placeholder="Your Name or Organization" className="rounded-lg" />
                )}
              </FormField>

              <FormField
                control={form.control}
                name="authorEmail"
                label={labelWithHint("Author Email (Optional)", "Contact email for the skill author")}
              >
                {({ ref, ...field }) => (
                  <Input {...field} ref={ref} type="email" placeholder="author@example.com" className="rounded-lg" />
                )}
              </FormField>
            </FieldGroup>

            <div className="mt-6 flex justify-end gap-2">
              <Button type="button" variant="outline" onClick={handleCancel} disabled={isSubmitting}>
                Cancel
              </Button>
              <Button type="submit" disabled={isSubmitting} aria-busy={isSubmitting}>
                {isSubmitting && <UiLoadingSpinner className="size-4" />}
                {isSubmitting ? "Adding..." : "Add Skill"}
              </Button>
            </div>
          </form>
        </TooltipProvider>
      </DialogContent>
    </Dialog>
  );
};

export default AddPluginForm;
