import React, { useState } from "react";
import { Upload } from "antd";
import type { UploadFile, UploadProps } from "antd";
import { Upload as UploadIcon } from "lucide-react";
import { z } from "zod/v4";
import { convertPromptFileToJson, createPromptCall } from "@/components/networking";
import { toast } from "@/lib/toast";
import { Field, FieldDescription, FieldGroup, FieldSeparator, FieldTitle } from "@/components/shared/form/field";
import { FormField } from "@/components/shared/form/FormField";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { useZodForm } from "@/lib/forms/useZodForm";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";

interface AddPromptFormProps {
  visible: boolean;
  onClose: () => void;
  accessToken: string | null;
  onSuccess: () => void;
}

interface CreatePromptRequest {
  prompt_id: string;
  litellm_params: {
    prompt_integration: string;
    prompt_id: string;
    prompt_data: unknown;
  };
  prompt_info: {
    prompt_type: string;
  };
}

const PROMPT_INTEGRATION_OPTIONS = [{ label: "dotprompt", value: "dotprompt" }];

const addPromptSchema = z.object({
  prompt_id: z
    .string()
    .min(1, "Please enter a prompt ID")
    .regex(/^[a-zA-Z0-9_-]+$/, "Prompt ID can only contain letters, numbers, underscores, and hyphens"),
  prompt_integration: z.string(),
});

type AddPromptFormValues = z.infer<typeof addPromptSchema>;

const EMPTY_VALUES: AddPromptFormValues = { prompt_id: "", prompt_integration: "dotprompt" };

const AddPromptForm: React.FC<AddPromptFormProps> = ({ visible, onClose, accessToken, onSuccess }) => {
  const form = useZodForm(addPromptSchema, { defaultValues: EMPTY_VALUES });
  const [loading, setLoading] = useState(false);
  const [fileList, setFileList] = useState<UploadFile[]>([]);
  const [promptIntegration, setPromptIntegration] = useState<string>("dotprompt");

  const handleCancel = () => {
    form.reset(EMPTY_VALUES);
    setFileList([]);
    setPromptIntegration("dotprompt");
    onClose();
  };

  const handleIntegrationChange = (selected: string | null) => {
    if (selected === null) return;
    form.setValue("prompt_integration", selected);
    setPromptIntegration(selected);
  };

  const convertUploadedFile = async (token: string, promptId: string): Promise<CreatePromptRequest | null> => {
    const file = fileList[0].originFileObj as File;

    try {
      const conversionResult = await convertPromptFileToJson(token, file);

      return {
        prompt_id: promptId,
        litellm_params: {
          prompt_integration: "dotprompt",
          prompt_id: conversionResult.prompt_id,
          prompt_data: conversionResult.json_data,
        },
        prompt_info: {
          prompt_type: "db",
        },
      };
    } catch (conversionError) {
      console.error("Error converting prompt file:", conversionError);
      toast.fromError("Failed to convert prompt file to JSON");
      return null;
    }
  };

  const handleSubmit = async (values: AddPromptFormValues) => {
    if (!accessToken) {
      toast.fromError("Access token is required");
      return;
    }

    const isDotprompt = promptIntegration === "dotprompt";

    if (isDotprompt && fileList.length === 0) {
      toast.fromError("Please upload a .prompt file");
      return;
    }

    setLoading(true);

    const promptData: CreatePromptRequest | Record<string, never> | null = isDotprompt
      ? await convertUploadedFile(accessToken, values.prompt_id)
      : {};

    if (promptData === null) {
      setLoading(false);
      return;
    }

    try {
      await createPromptCall(accessToken, promptData);
      toast.success("Prompt created successfully!");
      handleCancel();
      onSuccess();
    } catch (createError) {
      console.error("Error creating prompt:", createError);
      toast.fromError("Failed to create prompt");
    } finally {
      setLoading(false);
    }
  };

  const uploadProps: UploadProps = {
    beforeUpload: (file) => {
      if (!file.name.endsWith(".prompt")) {
        toast.fromError("Please upload a .prompt file");
        return false;
      }
      return false; // Prevent automatic upload
    },
    fileList,
    onChange: ({ fileList: newFileList }) => {
      setFileList(newFileList.slice(-1)); // Keep only the last file
    },
    onRemove: () => {
      setFileList([]);
    },
  };

  return (
    <Dialog open={visible} onOpenChange={(open) => !open && handleCancel()}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[600px]">
        <DialogHeader>
          <DialogTitle>Add New Prompt</DialogTitle>
        </DialogHeader>
        <form onSubmit={(event) => event.preventDefault()} noValidate>
          <FieldGroup>
            <FormField control={form.control} name="prompt_id" label="Prompt ID">
              {({ ref, ...field }) => (
                <Input {...field} ref={ref} placeholder="Enter unique prompt ID (e.g., my_prompt_id)" />
              )}
            </FormField>

            <FormField control={form.control} name="prompt_integration" label="Prompt Integration">
              {({ id, value, "aria-invalid": ariaInvalid, "aria-describedby": ariaDescribedBy }) => (
                <Select items={PROMPT_INTEGRATION_OPTIONS} value={value} onValueChange={handleIntegrationChange}>
                  <SelectTrigger id={id} aria-invalid={ariaInvalid} aria-describedby={ariaDescribedBy}>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {PROMPT_INTEGRATION_OPTIONS.map((option) => (
                      <SelectItem key={option.value} value={option.value}>
                        {option.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              )}
            </FormField>

            {promptIntegration === "dotprompt" && (
              <>
                <FieldSeparator />
                <Field>
                  <FieldTitle>Prompt File</FieldTitle>
                  <Upload {...uploadProps}>
                    <Button type="button" variant="outline">
                      <UploadIcon />
                      Select .prompt File
                    </Button>
                  </Upload>
                  {fileList.length > 0 && (
                    <div className="mt-2 text-sm text-muted-foreground">Selected: {fileList[0].name}</div>
                  )}
                  <FieldDescription>Upload a .prompt file that follows the Dotprompt specification</FieldDescription>
                </Field>
              </>
            )}
          </FieldGroup>
        </form>
        <DialogFooter>
          {" "}
          <Button type="button" variant="outline" onClick={handleCancel}>
            Cancel
          </Button>
          ,
          <Button type="button" disabled={loading} onClick={() => void form.handleSubmit(handleSubmit)()}>
            {loading && <UiLoadingSpinner className="size-4" />}
            Create Prompt
          </Button>
          , ]
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default AddPromptForm;
