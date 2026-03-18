/**
 * Modal wrapper for the fallback selection form
 * Handles modal visibility and layout, but delegates content to children
 */

import { Modal } from "antd";
import { ArrowRight } from "lucide-react";
import React from "react";

interface AddFallbacksModalProps {
  open: boolean;
  onCancel: () => void;
  children: React.ReactNode;
}

export function AddFallbacksModal({
  open,
  onCancel,
  children,
}: AddFallbacksModalProps) {
  return (
    <Modal
      title={
        <div className="pb-4 border-b border-gray-100">
          <div className="flex items-center gap-2 text-gray-800">
            <div className="p-2 bg-indigo-50 rounded-lg">
              <ArrowRight className="w-5 h-5 text-indigo-600" />
            </div>
            <div>
              <h2 className="text-lg font-bold m-0">Configure Model Fallbacks</h2>
              <p className="text-sm text-gray-500 font-normal m-0">
                Manage multiple fallback chains for different models (up to 5 groups at a time)
              </p>
            </div>
          </div>
        </div>
      }
      open={open}
      width={900}
      footer={null}
      onCancel={onCancel}
      maskClosable={false}
      className="top-8"
      styles={{
        body: { padding: "24px" },
        header: { padding: "24px 24px 0 24px", border: "none" },
      }}
    >
      <div className="mt-6">{children}</div>
    </Modal>
  );
}
