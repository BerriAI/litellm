import React from "react";
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
      onCancel={onCancel}
      closable={true}
      destroyOnClose={true}
      maskClosable={false}
    >
      <div className="mt-4">
        <p className="text-gray-900 mb-4">
          Deleting this credential may break existing model integrations that reference it. 
          Are you sure you want to delete the credential <strong>&ldquo;{credentialName}&rdquo;</strong>?
        </p>
        <p className="text-sm text-gray-600 mb-6">
          This action cannot be undone.
        </p>
        <div className="flex justify-end space-x-2">
          <TremorButton 
            onClick={onCancel} 
            variant="secondary"
            className="mr-2"
          >
            Cancel
          </TremorButton>
          <TremorButton
            onClick={onConfirm}
            color="red"
            className="focus:ring-red-500"
          >
            Delete Credential
          </TremorButton>
        </div>
      </div>
    </Modal>
  );
};

export default CredentialDeleteModal;