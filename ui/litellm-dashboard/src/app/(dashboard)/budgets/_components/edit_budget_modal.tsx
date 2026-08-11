import React, { useEffect } from "react";
import { TextInput, Accordion, AccordionHeader, AccordionBody } from "@tremor/react";
import { Button as Button2, Modal, Form, InputNumber, Select } from "antd";
import { useUpdateBudget } from "@/app/(dashboard)/hooks/budgets/useBudgets";
import { budgetItem } from "@/app/(dashboard)/hooks/budgets/useBudgets";
import NotificationsManager from "@/components/molecules/notifications_manager";
import { useTranslation } from "react-i18next";

interface EditBudgetModalProps {
  isModalVisible: boolean;
  setIsModalVisible: React.Dispatch<React.SetStateAction<boolean>>;
  existingBudget: budgetItem;
}
const EditBudgetModal: React.FC<EditBudgetModalProps> = ({ isModalVisible, setIsModalVisible, existingBudget }) => {
  const { t } = useTranslation("gateway");
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
      NotificationsManager.info(t("budgets.notifications.makingApiCall"));
      await updateBudget.mutateAsync(formValues);
      NotificationsManager.success(t("budgets.notifications.updated"));
      form.resetFields();
      setIsModalVisible(false);
    } catch (error) {
      console.error("Error updating the budget:", error);
      NotificationsManager.fromBackend(t("budgets.notifications.updateError", { error: String(error) }));
    }
  };

  return (
    <Modal
      title={t("budgets.form.editTitle")}
      open={isModalVisible}
      width={800}
      footer={null}
      onOk={handleOk}
      onCancel={handleCancel}
    >
      <Form
        form={form}
        onFinish={handleUpdate}
        labelCol={{ span: 8 }}
        wrapperCol={{ span: 16 }}
        labelAlign="left"
        initialValues={existingBudget}
      >
        <>
          <Form.Item label={t("budgets.fields.budgetId")} name="budget_id" help={t("budgets.form.budgetIdImmutable")}>
            <TextInput placeholder="" disabled={true} />
          </Form.Item>
          <Form.Item label={t("budgets.form.maxTpm")} name="tpm_limit" help={t("budgets.form.modelLimitDefault")}>
            <InputNumber step={1} precision={2} width={200} />
          </Form.Item>
          <Form.Item label={t("budgets.form.maxRpm")} name="rpm_limit" help={t("budgets.form.modelLimitDefault")}>
            <InputNumber step={1} precision={2} width={200} />
          </Form.Item>

          <Accordion className="mt-20 mb-8">
            <AccordionHeader>
              <b>{t("budgets.form.optional")}</b>
            </AccordionHeader>
            <AccordionBody>
              <Form.Item label={t("budgets.fields.maxBudgetUsd")} name="max_budget">
                <InputNumber step={0.01} precision={2} width={200} />
              </Form.Item>
              <Form.Item className="mt-8" label={t("budgets.form.resetBudget")} name="budget_duration">
                <Select defaultValue={null} placeholder={t("budgets.form.notSet")}>
                  <Select.Option value="24h">{t("budgets.duration.daily")}</Select.Option>
                  <Select.Option value="7d">{t("budgets.duration.weekly")}</Select.Option>
                  <Select.Option value="30d">{t("budgets.duration.monthly")}</Select.Option>
                </Select>
              </Form.Item>
            </AccordionBody>
          </Accordion>
        </>

        <div style={{ textAlign: "right", marginTop: "10px" }}>
          <Button2 htmlType="submit">{t("budgets.form.save")}</Button2>
        </div>
      </Form>
    </Modal>
  );
};

export default EditBudgetModal;
