import React from "react";
import { Modal } from "antd";
import { Button as TremorButton } from "@tremor/react";

interface DeleteModalProps {
  isVisible: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}

const DeleteModal: React.FC<DeleteModalProps> = ({ isVisible, onCancel, onConfirm }) => {
  return (
    <Modal title="Delete Vector Store" visible={isVisible} footer={null} onCancel={onCancel}>
      <p>Are you sure you want to delete this vector store? This action cannot be undone.</p>
      <div className="px-4 py-3 sm:px-6 sm:flex sm:flex-row-reverse">
        <TremorButton onClick={onConfirm} color="red" className="ml-2">
          Delete
        </TremorButton>
        <TremorButton onClick={onCancel} variant="primary">
          Cancel
        </TremorButton>
      </div>
    </Modal>
  );
};

export default DeleteModal;
