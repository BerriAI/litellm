import React from "react";
import { TextInput, Accordion, AccordionHeader, AccordionBody } from "@tremor/react";
import { Button as Button2, Modal, Form, InputNumber, Select } from "antd";
import { useTranslation } from "react-i18next";
import { useCreateBudget } from "@/app/(dashboard)/hooks/budgets/useBudgets";
import NotificationsManager from "../molecules/notifications_manager";

interface BudgetModalProps {
  isModalVisible: boolean;
  setIsModalVisible: React.Dispatch<React.SetStateAction<boolean>>;
}
const BudgetModal: React.FC<BudgetModalProps> = ({ isModalVisible, setIsModalVisible }) => {
  const { t } = useTranslation();
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
      NotificationsManager.info(t("budgets.budgetModal.makingApiCall"));
      await createBudget.mutateAsync(formValues);
      NotificationsManager.success(t("budgets.budgetModal.budgetCreated"));
      form.resetFields();
      setIsModalVisible(false);
    } catch (error) {
      console.error("Error creating the budget:", error);
      NotificationsManager.fromBackend(t("budgets.budgetModal.errorCreating", { error }));
    }
  };

  return (
    <Modal
      title={t("budgets.budgetModal.title")}
      open={isModalVisible}
      width={800}
      footer={null}
      onOk={handleOk}
      onCancel={handleCancel}
    >
      <Form form={form} onFinish={handleCreate} labelCol={{ span: 8 }} wrapperCol={{ span: 16 }} labelAlign="left">
        <>
          <Form.Item
            label={t("budgets.budgetModal.budgetIdLabel")}
            name="budget_id"
            rules={[
              {
                required: true,
                message: t("budgets.budgetModal.budgetIdRequired"),
              },
            ]}
            help={t("budgets.budgetModal.budgetIdHelp")}
          >
            <TextInput placeholder="" />
          </Form.Item>
          <Form.Item
            label={t("budgets.budgetModal.tpmLimitLabel")}
            name="tpm_limit"
            help={t("budgets.budgetModal.defaultModelLimit")}
          >
            <InputNumber step={1} precision={2} width={200} />
          </Form.Item>
          <Form.Item
            label={t("budgets.budgetModal.rpmLimitLabel")}
            name="rpm_limit"
            help={t("budgets.budgetModal.defaultModelLimit")}
          >
            <InputNumber step={1} precision={2} width={200} />
          </Form.Item>

          <Accordion className="mt-20 mb-8">
            <AccordionHeader>
              <b>{t("budgets.budgetModal.optionalSettings")}</b>
            </AccordionHeader>
            <AccordionBody>
              <Form.Item label={t("budgets.budgetModal.maxBudgetLabel")} name="max_budget">
                <InputNumber step={0.01} precision={2} width={200} />
              </Form.Item>
              <Form.Item className="mt-8" label={t("budgets.budgetModal.resetBudgetLabel")} name="budget_duration">
                <Select defaultValue={null} placeholder="n/a">
                  <Select.Option value="24h">{t("budgets.budgetModal.daily")}</Select.Option>
                  <Select.Option value="7d">{t("budgets.budgetModal.weekly")}</Select.Option>
                  <Select.Option value="30d">{t("budgets.budgetModal.monthly")}</Select.Option>
                </Select>
              </Form.Item>
            </AccordionBody>
          </Accordion>
        </>

        <div style={{ textAlign: "right", marginTop: "10px" }}>
          <Button2 htmlType="submit">{t("budgets.budgetModal.createBudgetButton")}</Button2>
        </div>
      </Form>
    </Modal>
  );
};

export default BudgetModal;
