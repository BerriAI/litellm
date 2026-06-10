import React, { useEffect } from "react";
import { TextInput, Accordion, AccordionHeader, AccordionBody } from "@tremor/react";
import { Button as Button2, Modal, Form, InputNumber, Select } from "antd";
import { useTranslation } from "react-i18next";
import { useUpdateBudget } from "@/app/(dashboard)/hooks/budgets/useBudgets";
import { budgetItem } from "./budget_panel";
import NotificationsManager from "../molecules/notifications_manager";

interface EditBudgetModalProps {
  isModalVisible: boolean;
  setIsModalVisible: React.Dispatch<React.SetStateAction<boolean>>;
  existingBudget: budgetItem;
}
const EditBudgetModal: React.FC<EditBudgetModalProps> = ({ isModalVisible, setIsModalVisible, existingBudget }) => {
  const { t } = useTranslation();
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
      NotificationsManager.info(t("budgets.editBudgetModal.makingApiCall"));
      await updateBudget.mutateAsync(formValues);
      NotificationsManager.success(t("budgets.editBudgetModal.budgetUpdated"));
      form.resetFields();
      setIsModalVisible(false);
    } catch (error) {
      console.error("Error updating the budget:", error);
      NotificationsManager.fromBackend(t("budgets.editBudgetModal.errorUpdating", { error }));
    }
  };

  return (
    <Modal
      title={t("budgets.editBudgetModal.title")}
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
          <Form.Item
            label={t("budgets.editBudgetModal.budgetIdLabel")}
            name="budget_id"
            help={t("budgets.editBudgetModal.budgetIdHelp")}
          >
            <TextInput placeholder="" disabled={true} />
          </Form.Item>
          <Form.Item
            label={t("budgets.editBudgetModal.tpmLimitLabel")}
            name="tpm_limit"
            help={t("budgets.editBudgetModal.defaultModelLimit")}
          >
            <InputNumber step={1} precision={2} width={200} />
          </Form.Item>
          <Form.Item
            label={t("budgets.editBudgetModal.rpmLimitLabel")}
            name="rpm_limit"
            help={t("budgets.editBudgetModal.defaultModelLimit")}
          >
            <InputNumber step={1} precision={2} width={200} />
          </Form.Item>

          <Accordion className="mt-20 mb-8">
            <AccordionHeader>
              <b>{t("budgets.editBudgetModal.optionalSettings")}</b>
            </AccordionHeader>
            <AccordionBody>
              <Form.Item label={t("budgets.editBudgetModal.maxBudgetLabel")} name="max_budget">
                <InputNumber step={0.01} precision={2} width={200} />
              </Form.Item>
              <Form.Item className="mt-8" label={t("budgets.editBudgetModal.resetBudgetLabel")} name="budget_duration">
                <Select defaultValue={null} placeholder="n/a">
                  <Select.Option value="24h">{t("budgets.editBudgetModal.daily")}</Select.Option>
                  <Select.Option value="7d">{t("budgets.editBudgetModal.weekly")}</Select.Option>
                  <Select.Option value="30d">{t("budgets.editBudgetModal.monthly")}</Select.Option>
                </Select>
              </Form.Item>
            </AccordionBody>
          </Accordion>
        </>

        <div style={{ textAlign: "right", marginTop: "10px" }}>
          <Button2 htmlType="submit">{t("common.save")}</Button2>
        </div>
      </Form>
    </Modal>
  );
};

export default EditBudgetModal;
