import { Button, Form, Modal, Tooltip, Typography } from "antd";
import { useState } from "react";
import ProviderSpecificFields from "./add_model/provider_specific_fields";
import { modelPatchUpdateCall } from "./networking";
import NotificationsManager from "./molecules/notifications_manager";
import { Providers } from "./provider_info_helpers";

const { Link, Text } = Typography;

interface UpdateModelCredentialsModalProps {
  open: boolean;
  onCancel: () => void;
  accessToken: string;
  modelId: string;
  provider: Providers;
  onUpdated: () => void;
}

const filterFilledFields = (values: Record<string, unknown>): Record<string, unknown> =>
  Object.fromEntries(
    Object.entries(values).filter(([, value]) => value !== "" && value !== undefined && value !== null),
  );

export default function UpdateModelCredentialsModal({
  open,
  onCancel,
  accessToken,
  modelId,
  provider,
  onUpdated,
}: UpdateModelCredentialsModalProps) {
  const [form] = Form.useForm();
  const [isSaving, setIsSaving] = useState(false);

  const close = () => {
    form.resetFields();
    onCancel();
  };

  const handleSubmit = async (values: Record<string, unknown>) => {
    const filled = filterFilledFields(values);
    if (Object.keys(filled).length === 0) {
      NotificationsManager.fromBackend("Enter at least one field to update");
      return;
    }
    setIsSaving(true);
    try {
      await modelPatchUpdateCall(accessToken, { litellm_params: filled, model_info: { id: modelId } }, modelId);
      NotificationsManager.success("Credentials updated");
      form.resetFields();
      onUpdated();
      onCancel();
    } catch (error) {
      console.error("Error updating credentials:", error);
      NotificationsManager.fromBackend("Failed to update credentials");
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <Modal title="Update Credentials" open={open} onCancel={close} footer={null} width={600} destroyOnHidden={true}>
      <Text className="block mb-4 text-gray-500">
        Enter a new value for any field you want to rotate. Only the fields you fill in are sent; everything else on the
        model is left untouched. Leave a field blank to keep its current value.
      </Text>
      <Form form={form} onFinish={handleSubmit} layout="vertical">
        <ProviderSpecificFields selectedProvider={provider} disableRequired />
        <div className="flex justify-between items-center mt-4">
          <Tooltip title="Get help on our github">
            <Link href="https://github.com/BerriAI/litellm/issues">Need Help?</Link>
          </Tooltip>
          <div>
            <Button onClick={close} style={{ marginRight: 10 }}>
              Cancel
            </Button>
            <Button htmlType="submit" loading={isSaving}>
              Update Credentials
            </Button>
          </div>
        </div>
      </Form>
    </Modal>
  );
}
