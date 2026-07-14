import React from "react";
import { Modal, Form, DatePicker, Button as Button2 } from "antd";
import { PtuReservationItem, useClosePtuReservation } from "@/app/(dashboard)/hooks/ptuReservations/usePtuReservations";
import NotificationsManager from "@/components/molecules/notifications_manager";

interface CloseReservationModalProps {
  isModalVisible: boolean;
  setIsModalVisible: React.Dispatch<React.SetStateAction<boolean>>;
  reservation: PtuReservationItem | null;
}

const CloseReservationModal: React.FC<CloseReservationModalProps> = ({
  isModalVisible,
  setIsModalVisible,
  reservation,
}) => {
  const [form] = Form.useForm();
  const closeReservation = useClosePtuReservation();

  const handleCancel = () => {
    setIsModalVisible(false);
    form.resetFields();
  };

  const handleClose = async (formValues: Record<string, any>) => {
    if (!reservation) return;
    try {
      NotificationsManager.info("Closing reservation");
      await closeReservation.mutateAsync({
        id: reservation.id,
        effective_to: formValues.effective_to?.toISOString(),
      });
      NotificationsManager.success("Reservation closed");
      form.resetFields();
      setIsModalVisible(false);
    } catch (error) {
      console.error("Error closing PTU reservation:", error);
      NotificationsManager.fromBackend(`Error closing reservation: ${error}`);
    }
  };

  return (
    <Modal
      title="Close PTU Reservation"
      open={isModalVisible}
      width={640}
      footer={null}
      onCancel={handleCancel}
      destroyOnHidden
    >
      <p>
        Closing sets <code>effective_to</code> so the reservation stops accruing daily flat cost. Historical rollup rows
        already written are preserved for audit.
      </p>
      <Form
        form={form}
        onFinish={handleClose}
        labelCol={{ span: 8 }}
        wrapperCol={{ span: 16 }}
        labelAlign="left"
        style={{ marginTop: 16 }}
      >
        <Form.Item label="Effective to" name="effective_to" help="Optional; defaults to now (UTC)">
          <DatePicker showTime style={{ width: "100%" }} />
        </Form.Item>

        <div style={{ textAlign: "right", marginTop: "10px" }}>
          <Button2 htmlType="submit" loading={closeReservation.isPending} danger>
            Close reservation
          </Button2>
        </div>
      </Form>
    </Modal>
  );
};

export default CloseReservationModal;
