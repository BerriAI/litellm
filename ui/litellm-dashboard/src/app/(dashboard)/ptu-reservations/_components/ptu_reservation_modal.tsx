import React from "react";
import { Button as Button2, Modal, Form, Input, InputNumber, DatePicker } from "antd";
import { useCreatePtuReservation } from "@/app/(dashboard)/hooks/ptuReservations/usePtuReservations";
import NotificationsManager from "@/components/molecules/notifications_manager";

interface PtuReservationModalProps {
  isModalVisible: boolean;
  setIsModalVisible: React.Dispatch<React.SetStateAction<boolean>>;
}

const PtuReservationModal: React.FC<PtuReservationModalProps> = ({ isModalVisible, setIsModalVisible }) => {
  const [form] = Form.useForm();
  const createReservation = useCreatePtuReservation();

  const handleCancel = () => {
    setIsModalVisible(false);
    form.resetFields();
  };

  const handleCreate = async (formValues: Record<string, any>) => {
    try {
      const body: Record<string, any> = {
        team_id: formValues.team_id,
        model: formValues.model,
        ptu_count: formValues.ptu_count,
        cost_per_ptu: formValues.cost_per_ptu,
        effective_from: formValues.effective_from?.toISOString(),
      };
      if (formValues.effective_to) {
        body.effective_to = formValues.effective_to.toISOString();
      }
      NotificationsManager.info("Creating reservation");
      await createReservation.mutateAsync(body);
      NotificationsManager.success("PTU reservation created");
      form.resetFields();
      setIsModalVisible(false);
    } catch (error) {
      console.error("Error creating PTU reservation:", error);
      NotificationsManager.fromBackend(`Error creating PTU reservation: ${error}`);
    }
  };

  return (
    <Modal
      title="Create PTU Reservation"
      open={isModalVisible}
      width={720}
      footer={null}
      onCancel={handleCancel}
      destroyOnHidden
    >
      <Form form={form} onFinish={handleCreate} labelCol={{ span: 8 }} wrapperCol={{ span: 16 }} labelAlign="left">
        <Form.Item
          label="Team ID"
          name="team_id"
          rules={[{ required: true, message: "Team id is required" }]}
          help="Team that owns the reservation"
        >
          <Input placeholder="e.g., team-x" />
        </Form.Item>
        <Form.Item
          label="Model"
          name="model"
          rules={[{ required: true, message: "Model is required" }]}
          help="Model this reservation covers (e.g. gpt-4)"
        >
          <Input placeholder="e.g., gpt-4" />
        </Form.Item>
        <Form.Item
          label="PTU count"
          name="ptu_count"
          rules={[
            { required: true, message: "PTU count is required" },
            { type: "number", min: 1, message: "PTU count must be at least 1" },
          ]}
        >
          <InputNumber step={1} min={1} precision={0} style={{ width: 200 }} />
        </Form.Item>
        <Form.Item
          label="Cost per PTU (USD/month)"
          name="cost_per_ptu"
          rules={[
            { required: true, message: "Cost per PTU is required" },
            { type: "number", min: 0, message: "Cost cannot be negative" },
          ]}
        >
          <InputNumber step={0.01} min={0} precision={2} style={{ width: 200 }} />
        </Form.Item>
        <Form.Item
          label="Effective from"
          name="effective_from"
          rules={[{ required: true, message: "Effective from is required" }]}
          help="Inclusive UTC start"
        >
          <DatePicker showTime style={{ width: "100%" }} />
        </Form.Item>
        <Form.Item label="Effective to" name="effective_to" help="Optional; leave empty for still active">
          <DatePicker showTime style={{ width: "100%" }} />
        </Form.Item>

        <div style={{ textAlign: "right", marginTop: "10px" }}>
          <Button2 htmlType="submit" loading={createReservation.isPending}>
            Create
          </Button2>
        </div>
      </Form>
    </Modal>
  );
};

export default PtuReservationModal;
