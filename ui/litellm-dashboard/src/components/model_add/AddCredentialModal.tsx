import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Select as AntdSelect,
  Form,
} from "antd";
import type { UploadProps } from "antd/es/upload";
import React, { useState } from "react";
import ProviderSpecificFields from "../add_model/provider_specific_fields";
import { Providers, providerLogoMap } from "../provider_info_helpers";

interface AddCredentialsModalProps {
  open: boolean;
  onCancel: () => void;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  onAddCredential: (values: any) => void;
  uploadProps: UploadProps;
}

const AddCredentialsModal: React.FC<AddCredentialsModalProps> = ({
  open,
  onCancel,
  onAddCredential,
  uploadProps,
}) => {
  const [form] = Form.useForm();
  const [selectedProvider, setSelectedProvider] = useState<Providers>(
    Providers.OpenAI,
  );

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const handleSubmit = (values: any) => {
    const filteredValues = Object.entries(values).reduce(
      (acc, [key, value]) => {
        if (value !== "" && value !== undefined && value !== null) {
          acc[key] = value;
        }
        return acc;
      },
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      {} as any,
    );
    onAddCredential(filteredValues);
    form.resetFields();
  };

  const handleCancel = () => {
    onCancel();
    form.resetFields();
  };

  return (
    <Dialog open={open} onOpenChange={(o) => (!o ? handleCancel() : undefined)}>
      <DialogContent className="max-w-[600px]">
        <DialogHeader>
          <DialogTitle>Add New Credential</DialogTitle>
        </DialogHeader>
        <Form form={form} onFinish={handleSubmit} layout="vertical">
          <Form.Item
            label="Credential Name:"
            name="credential_name"
            rules={[{ required: true, message: "Credential name is required" }]}
          >
            <Input placeholder="Enter a friendly name for these credentials" />
          </Form.Item>

          <Form.Item
            rules={[{ required: true, message: "Required" }]}
            label="Provider:"
            name="custom_llm_provider"
            tooltip="Helper to auto-populate provider specific fields"
          >
            <AntdSelect
              showSearch
              onChange={(value) => {
                setSelectedProvider(value as Providers);
                form.setFieldValue("custom_llm_provider", value);
              }}
            >
              {Object.entries(Providers).map(
                ([providerEnum, providerDisplayName]) => (
                  <AntdSelect.Option key={providerEnum} value={providerEnum}>
                    <div className="flex items-center space-x-2">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={providerLogoMap[providerDisplayName]}
                        alt={`${providerEnum} logo`}
                        className="w-5 h-5"
                        onError={(e) => {
                          const target = e.target as HTMLImageElement;
                          const parent = target.parentElement;
                          if (parent) {
                            const fallbackDiv = document.createElement("div");
                            fallbackDiv.className =
                              "w-5 h-5 rounded-full bg-muted flex items-center justify-center text-xs";
                            fallbackDiv.textContent =
                              providerDisplayName.charAt(0);
                            parent.replaceChild(fallbackDiv, target);
                          }
                        }}
                      />
                      <span>{providerDisplayName}</span>
                    </div>
                  </AntdSelect.Option>
                ),
              )}
            </AntdSelect>
          </Form.Item>

          <ProviderSpecificFields
            selectedProvider={selectedProvider}
            uploadProps={uploadProps}
          />

          <div className="flex justify-between items-center">
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <a
                    href="https://github.com/BerriAI/litellm/issues"
                    className="text-primary hover:underline"
                  >
                    Need Help?
                  </a>
                </TooltipTrigger>
                <TooltipContent>Get help on our github</TooltipContent>
              </Tooltip>
            </TooltipProvider>

            <div className="flex gap-2">
              <Button variant="outline" onClick={handleCancel}>
                Cancel
              </Button>
              <Button type="submit">Add Credential</Button>
            </div>
          </div>
        </Form>
      </DialogContent>
    </Dialog>
  );
};

export default AddCredentialsModal;
