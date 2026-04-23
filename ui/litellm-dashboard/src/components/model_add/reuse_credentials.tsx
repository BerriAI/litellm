import React from "react";
import { Form } from "antd";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { CredentialItem } from "../networking";

interface ReuseCredentialsModalProps {
  isVisible: boolean;
  onCancel: () => void;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  onAddCredential: (values: any) => void;
  existingCredential: CredentialItem | null;
  setIsCredentialModalOpen: (isVisible: boolean) => void;
}

const ReuseCredentialsModal: React.FC<ReuseCredentialsModalProps> = ({
  isVisible,
  onCancel,
  onAddCredential,
  existingCredential,
  setIsCredentialModalOpen,
}) => {
  const [form] = Form.useForm();

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const handleSubmit = (values: any) => {
    onAddCredential(values);
    form.resetFields();
    setIsCredentialModalOpen(false);
  };

  return (
    <Dialog
      open={isVisible}
      onOpenChange={(o) => {
        if (!o) {
          onCancel();
          form.resetFields();
        }
      }}
    >
      <DialogContent className="max-w-[600px]">
        <DialogHeader>
          <DialogTitle>Reuse Credentials</DialogTitle>
        </DialogHeader>
        <Form form={form} onFinish={handleSubmit} layout="vertical">
          {/* Credential Name */}
          <Form.Item
            label="Credential Name:"
            name="credential_name"
            rules={[
              { required: true, message: "Credential name is required" },
            ]}
            initialValue={existingCredential?.credential_name}
          >
            <Input placeholder="Enter a friendly name for these credentials" />
          </Form.Item>

          {/* Display Credential Values of existingCredential, don't allow user to edit. */}
          {Object.entries(existingCredential?.credential_values || {}).map(
            ([key, value]) => (
              <Form.Item key={key} label={key} name={key} initialValue={value}>
                <Input placeholder={`Enter ${key}`} disabled={true} />
              </Form.Item>
            ),
          )}

          <div className="flex justify-between items-center">
            <a
              href="https://github.com/BerriAI/litellm/issues"
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary hover:text-primary/80 text-sm"
              title="Get help on our github"
            >
              Need Help?
            </a>

            <div className="flex gap-2">
              <Button
                type="button"
                variant="outline"
                onClick={() => {
                  onCancel();
                  form.resetFields();
                }}
              >
                Cancel
              </Button>
              <Button type="submit">Reuse Credentials</Button>
            </div>
          </div>
        </Form>
      </DialogContent>
    </Dialog>
  );
};

export default ReuseCredentialsModal;
