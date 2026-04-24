import React, { useRef, useState } from "react";
import { Controller, FormProvider, useForm } from "react-hook-form";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogFooter,
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
import { Upload as UploadIcon, X } from "lucide-react";
import { convertPromptFileToJson, createPromptCall } from "../networking";
import NotificationsManager from "../molecules/notifications_manager";

interface AddPromptFormProps {
  visible: boolean;
  onClose: () => void;
  accessToken: string | null;
  onSuccess: () => void;
}

interface PromptFormValues {
  prompt_id: string;
  prompt_integration: string;
}

const PROMPT_ID_PATTERN = /^[a-zA-Z0-9_-]+$/;

const AddPromptForm: React.FC<AddPromptFormProps> = ({ visible, onClose, accessToken, onSuccess }) => {
  const [loading, setLoading] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const form = useForm<PromptFormValues>({
    defaultValues: {
      prompt_id: "",
      prompt_integration: "dotprompt",
    },
  });

  const promptIntegration = form.watch("prompt_integration");

  const handleCancel = () => {
    form.reset({ prompt_id: "", prompt_integration: "dotprompt" });
    setSelectedFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
    onClose();
  };

  const onSubmit = form.handleSubmit(async (values) => {
    console.log("values: ", values);
    if (!accessToken) {
      NotificationsManager.fromBackend("Access token is required");
      return;
    }

    if (values.prompt_integration === "dotprompt" && !selectedFile) {
      NotificationsManager.fromBackend("Please upload a .prompt file");
      return;
    }

    setLoading(true);

    let promptData: any = {};

    if (values.prompt_integration === "dotprompt" && selectedFile) {
      try {
        const conversionResult = await convertPromptFileToJson(accessToken, selectedFile);
        console.log("Conversion result:", conversionResult);

        promptData = {
          prompt_id: values.prompt_id,
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
        NotificationsManager.fromBackend("Failed to convert prompt file to JSON");
        setLoading(false);
        return;
      }
    }

    try {
      await createPromptCall(accessToken, promptData);
      NotificationsManager.success("Prompt created successfully!");
      handleCancel();
      onSuccess();
    } catch (createError) {
      console.error("Error creating prompt:", createError);
      NotificationsManager.fromBackend("Failed to create prompt");
    } finally {
      setLoading(false);
    }
  });

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) {
      setSelectedFile(null);
      return;
    }
    if (!file.name.endsWith(".prompt")) {
      NotificationsManager.fromBackend("Please upload a .prompt file");
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
      setSelectedFile(null);
      return;
    }
    setSelectedFile(file);
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  return (
    <Dialog open={visible} onOpenChange={(o) => (!o ? handleCancel() : undefined)}>
      <DialogContent className="max-w-[600px]">
        <DialogHeader>
          <DialogTitle>Add New Prompt</DialogTitle>
        </DialogHeader>
        <FormProvider {...form}>
          <form onSubmit={onSubmit} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="prompt_id">
                Prompt ID <span className="text-destructive">*</span>
              </Label>
              <Input
                id="prompt_id"
                placeholder="Enter unique prompt ID (e.g., my_prompt_id)"
                {...form.register("prompt_id", {
                  required: "Please enter a prompt ID",
                  pattern: {
                    value: PROMPT_ID_PATTERN,
                    message:
                      "Prompt ID can only contain letters, numbers, underscores, and hyphens",
                  },
                })}
              />
              {form.formState.errors.prompt_id && (
                <p className="text-sm text-destructive">
                  {form.formState.errors.prompt_id.message as string}
                </p>
              )}
            </div>

            <div className="space-y-2">
              <Label htmlFor="prompt_integration">Prompt Integration</Label>
              <Controller
                control={form.control}
                name="prompt_integration"
                render={({ field }) => (
                  <Select value={field.value} onValueChange={field.onChange}>
                    <SelectTrigger id="prompt_integration">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="dotprompt">dotprompt</SelectItem>
                    </SelectContent>
                  </Select>
                )}
              />
            </div>

            {promptIntegration === "dotprompt" && (
              <>
                <hr className="my-4 border-border" />
                <div className="space-y-2">
                  <Label>Prompt File</Label>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".prompt"
                    className="hidden"
                    onChange={handleFileChange}
                  />
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <UploadIcon className="h-4 w-4" />
                    Select .prompt File
                  </Button>
                  <p className="text-xs text-muted-foreground">
                    Upload a .prompt file that follows the Dotprompt specification
                  </p>
                  {selectedFile && (
                    <div className="mt-2 flex items-center gap-2 text-sm text-muted-foreground">
                      <span>Selected: {selectedFile.name}</span>
                      <button
                        type="button"
                        onClick={handleRemoveFile}
                        className="text-muted-foreground hover:text-foreground"
                        aria-label="Remove file"
                      >
                        <X className="h-3 w-3" />
                      </button>
                    </div>
                  )}
                </div>
              </>
            )}

            <DialogFooter>
              <Button type="button" variant="outline" onClick={handleCancel}>
                Cancel
              </Button>
              <Button type="submit" disabled={loading}>
                {loading ? "Creating…" : "Create Prompt"}
              </Button>
            </DialogFooter>
          </form>
        </FormProvider>
      </DialogContent>
    </Dialog>
  );
};

export default AddPromptForm;
