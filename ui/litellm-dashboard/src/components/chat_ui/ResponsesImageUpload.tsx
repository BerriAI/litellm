import React from "react";
import { Upload, Tooltip } from "antd";
import { PaperClipOutlined } from "@ant-design/icons";

const { Dragger } = Upload;

interface ResponsesImageUploadProps {
  responsesUploadedImage: File | null;
  responsesImagePreviewUrl: string | null;
  onImageUpload: (file: File) => false;
  onRemoveImage: () => void;
}

const ResponsesImageUpload: React.FC<ResponsesImageUploadProps> = ({
  responsesUploadedImage,
  responsesImagePreviewUrl,
  onImageUpload,
  onRemoveImage,
}) => {
  return (
    <>
      {/* Subtle upload button - only show when no image */}
      {!responsesUploadedImage && (
        <Dragger
          beforeUpload={onImageUpload}
          accept="image/*,.pdf"
          showUploadList={false}
          className="inline-block"
          style={{ padding: 0, border: "none", background: "none" }}
        >
          <Tooltip title="Attach image or PDF">
            <button
              type="button"
              className="flex items-center justify-center w-8 h-8 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-md transition-colors"
            >
              <PaperClipOutlined style={{ fontSize: "16px" }} />
            </button>
          </Tooltip>
        </Dragger>
      )}
    </>
  );
};

export default ResponsesImageUpload;
