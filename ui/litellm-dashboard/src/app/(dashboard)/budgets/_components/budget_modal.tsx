import React from "react";
import { ChevronDown } from "lucide-react";
import { Button as Button2, Modal, Form, InputNumber, Select } from "antd";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { useCreateBudget } from "@/app/(dashboard)/hooks/budgets/useBudgets";
import NotificationsManager from "@/components/molecules/notifications_manager";

interface BudgetModalProps {
  isModalVisible: boolean;
  setIsModalVisible: React.Dispatch<React.SetStateAction<boolean>>;
}
const BudgetModal: React.FC<BudgetModalProps> = ({ isModalVisible, setIsModalVisible }) => {
  const [form] = Form.useForm();
  const createBudget = useCreateBudget();

  const handleOk = () => {
    setIsModalVisible(false);
    form.resetFields();
  };

  const handleCancel = () => {
    setIsModalVisible(false);
    form.resetFields();
  };

  const handleCreate = async (formValues: Record<string, any>) => {
    try {
      NotificationsManager.info("Making API Call");
      await createBudget.mutateAsync(formValues);
      NotificationsManager.success("Budget Created");
      form.resetFields();
      setIsModalVisible(false);
    } catch (error) {
      console.error("Error creating the budget:", error);
      NotificationsManager.fromBackend(`Error creating the budget: ${error}`);
    }
  };

  return (
    <Modal
      title="Create Budget"
      open={isModalVisible}
      width={800}
      footer={null}
      onOk={handleOk}
      onCancel={handleCancel}
    >
      <Form form={form} onFinish={handleCreate} labelCol={{ span: 8 }} wrapperCol={{ span: 16 }} labelAlign="left">
        <>
          <Form.Item
            label="Budget ID"
            name="budget_id"
            rules={[
              {
                required: true,
                message: "Please input a human-friendly name for the budget",
              },
            ]}
            help="A human-friendly name for the budget"
          >
            <Input />
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
          <Button2 htmlType="submit">Create Budget</Button2>
        </div>
      </Form>
    </Modal>
  );
};

export default BudgetModal;
