import React, { useEffect } from "react";
import { ChevronDown } from "lucide-react";
import { Button as Button2, Modal, Form, InputNumber, Select } from "antd";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { useUpdateBudget } from "@/app/(dashboard)/hooks/budgets/useBudgets";
import { budgetItem } from "@/app/(dashboard)/hooks/budgets/useBudgets";
import NotificationsManager from "@/components/molecules/notifications_manager";

interface EditBudgetModalProps {
  isModalVisible: boolean;
  setIsModalVisible: React.Dispatch<React.SetStateAction<boolean>>;
  existingBudget: budgetItem;
}
const EditBudgetModal: React.FC<EditBudgetModalProps> = ({ isModalVisible, setIsModalVisible, existingBudget }) => {
  const [form] = Form.useForm();
  const updateBudget = useUpdateBudget();

  useEffect(() => {
    form.setFieldsValue(existingBudget);
  }, [existingBudget, form]);

  const handleOk = () => {
    setIsModalVisible(false);
    form.resetFields();
  };

  const handleCancel = () => {
    setIsModalVisible(false);
    form.resetFields();
  };

  const handleUpdate = async (formValues: Record<string, any>) => {
    try {
      NotificationsManager.info("Making API Call");
      await updateBudget.mutateAsync(formValues);
      NotificationsManager.success("Budget Updated");
      form.resetFields();
      setIsModalVisible(false);
    } catch (error) {
      console.error("Error updating the budget:", error);
      NotificationsManager.fromBackend(`Error updating the budget: ${error}`);
    }
  };

  return (
    <Modal title="Edit Budget" open={isModalVisible} width={800} footer={null} onOk={handleOk} onCancel={handleCancel}>
      <Form
        form={form}
        onFinish={handleUpdate}
        labelCol={{ span: 8 }}
        wrapperCol={{ span: 16 }}
        labelAlign="left"
        initialValues={existingBudget}
      >
        <>
          <Form.Item label="Budget ID" name="budget_id" help="Budget ID cannot be changed after creation">
            <Input placeholder="" disabled={true} />
          </Form.Item>
          <Form.Item label="Max Tokens per minute" name="tpm_limit" help="Default is model limit.">
            <InputNumber step={1} precision={2} width={200} />
          </Form.Item>
          <Form.Item label="Max Requests per minute" name="rpm_limit" help="Default is model limit.">
            <InputNumber step={1} precision={2} width={200} />
          </Form.Item>

          <Collapsible className="mt-20 mb-8 overflow-hidden rounded-lg border">
            <CollapsibleTrigger className="group/section flex w-full items-center justify-between px-4 py-3 text-left">
              <b>Optional Settings</b>
              <ChevronDown className="size-5 shrink-0 text-gray-500 transition-transform group-data-[panel-open]/section:rotate-180" />
            </CollapsibleTrigger>
            <CollapsibleContent className="px-4 pb-3">
              <Form.Item label="Max Budget (USD)" name="max_budget">
                <InputNumber step={0.01} precision={2} width={200} />
              </Form.Item>
              <Form.Item className="mt-8" label="Reset Budget" name="budget_duration">
                <Select defaultValue={null} placeholder="n/a">
                  <Select.Option value="24h">daily</Select.Option>
                  <Select.Option value="7d">weekly</Select.Option>
                  <Select.Option value="30d">monthly</Select.Option>
                </Select>
              </Form.Item>
            </CollapsibleContent>
          </Collapsible>
        </>

        <div style={{ textAlign: "right", marginTop: "10px" }}>
          <Button2 htmlType="submit">Save</Button2>
        </div>
      </Form>
    </Modal>
  );
};

export default EditBudgetModal;
