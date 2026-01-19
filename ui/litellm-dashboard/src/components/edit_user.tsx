import { useEffect, useState } from "react";
import { TextInput, SelectItem } from "@tremor/react";

import { Button as Button2, Modal, Form, Select as Select2, InputNumber } from "antd";

import NumericalInput from "./shared/numerical_input";
import BudgetDurationDropdown from "./common_components/budget_duration_dropdown";

interface EditUserModalProps {
  visible: boolean;
  possibleUIRoles: null | Record<string, Record<string, string>>;
  onCancel: () => void;
  user: any;
  onSubmit: (data: any) => void;
}

const EditUserModal: React.FC<EditUserModalProps> = ({ visible, possibleUIRoles, onCancel, user, onSubmit }) => {
  const [editedUser, setEditedUser] = useState(user);
  const [form] = Form.useForm();

  useEffect(() => {
    form.resetFields();
  }, [user]);

  const handleCancel = async () => {
    form.resetFields();
    onCancel();
  };

  const handleEditSubmit = async (formValues: Record<string, any>) => {
    // Call API to update team with teamId and values
    onSubmit(formValues);
    form.resetFields();
    onCancel();
  };

  if (!user) {
    return null;
  }

  return (
    <Modal
      visible={visible}
      onCancel={handleCancel}
      footer={null}
      title={"Редактировать пользователя " + user.user_id}
      width={1000}
    >
      <Form
        form={form}
        onFinish={handleEditSubmit}
        initialValues={user} // Pass initial values here
        labelCol={{ span: 8 }}
        wrapperCol={{ span: 16 }}
        labelAlign="left"
      >
        <>
          <Form.Item className="mt-8" label="Почта пользователя" tooltip="Почта пользователя" name="user_email">
            <TextInput />
          </Form.Item>

          <Form.Item label="user_id" name="user_id" hidden={true}>
            <TextInput />
          </Form.Item>

          <Form.Item label="Роль пользователя" name="user_role">
            <Select2>
              {possibleUIRoles &&
                Object.entries(possibleUIRoles).map(([role, { ui_label, description }]) => (
                  <SelectItem key={role} value={role} title={ui_label}>
                    <div className="flex">
                      {ui_label}{" "}
                      <p className="ml-2" style={{ color: "gray", fontSize: "12px" }}>
                        {description}
                      </p>
                    </div>
                  </SelectItem>
                ))}
            </Select2>
          </Form.Item>

          <Form.Item
            label="Расходы (USD)"
            name="spend"
            tooltip="(float) - Расходы на все завершённые вызовы LLM этого пользователя"
            help="По всем ключам (включая ключи с team_id)."
          >
            <InputNumber min={0} step={0.01} />
          </Form.Item>

          <Form.Item
            label="Бюджет пользователя (USD)"
            name="max_budget"
            tooltip="(float) - Максимальный бюджет этого пользователя"
            help="Максимальный бюджет этого пользователя."
          >
            <NumericalInput min={0} step={0.01} />
          </Form.Item>

          <Form.Item label="Сбросить бюджет" name="budget_duration">
            <BudgetDurationDropdown />
          </Form.Item>

          <div style={{ textAlign: "right", marginTop: "10px" }}>
            <Button2 htmlType="submit">Сохранить</Button2>
          </div>

          <div style={{ textAlign: "right", marginTop: "10px" }}>
            <Button2 htmlType="submit">Сохранить</Button2>
          </div>
        </>
      </Form>
    </Modal>
  );
};

export default EditUserModal;
