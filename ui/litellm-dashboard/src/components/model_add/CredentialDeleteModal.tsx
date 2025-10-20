import React, { useState } from "react";
import { Modal } from "antd";
import { Button as TremorButton } from "@tremor/react";
import { ExclamationIcon } from "@heroicons/react/outline";

interface CredentialDeleteModalProps {
  isVisible: boolean;
  onCancel: () => void;
  onConfirm: () => void;
  credentialName: string;
}

const CredentialDeleteModal: React.FC<CredentialDeleteModalProps> = ({
  isVisible,
  onCancel,
  onConfirm,
  credentialName,
}) => {
  const [deleteConfirmInput, setDeleteConfirmInput] = useState("");
  const isValid = deleteConfirmInput === credentialName;

  const handleCancel = () => {
    setDeleteConfirmInput("");
    onCancel();
  };

  const handleConfirm = () => {
    if (isValid) {
      setDeleteConfirmInput("");
      onConfirm();
    }
  };

  return (
    <Modal
      title={
        <div className="flex items-center">
          <ExclamationIcon className="h-6 w-6 text-red-600 mr-2" />
          Delete Credential
        </div>
      }
      open={isVisible}
      footer={null}
      onCancel={handleCancel}
      closable={true}
      destroyOnClose={true}
      maskClosable={false}
    >
      <div className="mt-4">
        <div className="flex items-start gap-3 p-4 bg-red-50 border border-red-100 rounded-md mb-5">
          <div className="text-red-500 mt-0.5">
            <ExclamationIcon className="h-5 w-5" />
          </div>
          <div>
            <p className="text-base font-medium text-red-600">
              This action cannot be undone and may break existing integrations.
            </p>
          </div>
        </div>

        <div className="mb-5">
          <label className="block text-base font-medium text-gray-700 mb-2">
            {`Type `}
            <span className="underline italic">&apos;{credentialName}&apos;</span>
            {` to confirm deletion:`}
          </label>
          <input
            type="text"
            value={deleteConfirmInput}
            onChange={(e) => setDeleteConfirmInput(e.target.value)}
            placeholder="Enter credential name exactly"
            className="w-full px-4 py-3 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-base"
            autoFocus
          />
        </div>

        <div className="flex justify-end space-x-2">
          <TremorButton onClick={handleCancel} variant="secondary" className="mr-2">
            Cancel
          </TremorButton>
          <TremorButton onClick={handleConfirm} color="red" className="focus:ring-red-500" disabled={!isValid}>
            Delete Credential
          </TremorButton>
        </div>
      </div>
    </Modal>
  );
};

export default CredentialDeleteModal;
