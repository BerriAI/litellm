import React from "react";
import { Modal } from "antd";
import { AlertTriangle } from "lucide-react";

interface DeleteCustomerModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  customerName: string;
  customerId: string;
}

const DeleteCustomerModal: React.FC<DeleteCustomerModalProps> = ({
  isOpen,
  onClose,
  onConfirm,
  customerName,
  customerId,
}) => {
  return (
    <Modal
      title={
        <div className="flex items-center gap-2">
          <AlertTriangle className="w-5 h-5 text-red-600" />
          <span>Delete Customer</span>
        </div>
      }
      open={isOpen}
      onOk={onConfirm}
      onCancel={onClose}
      okText="Delete"
      okButtonProps={{ danger: true }}
      cancelText="Cancel"
    >
      <div className="py-4">
        <p className="text-sm text-gray-900 mb-2">
          Are you sure you want to delete this customer?
        </p>
        <p className="text-sm text-gray-500">
          This will permanently delete{" "}
          <span className="font-medium text-gray-700">
            {customerName || customerId}
          </span>{" "}
          <span className="text-gray-400">({customerId})</span>. This action
          cannot be undone.
        </p>
      </div>
    </Modal>
  );
};

export default DeleteCustomerModal;
