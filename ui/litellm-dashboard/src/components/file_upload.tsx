import React from 'react';
import { Upload, message } from 'antd';
import { UploadOutlined } from '@ant-design/icons';
import type { UploadProps } from 'antd';

interface FileUploadProps {
  onFileUploaded?: (content: string) => void;
  accept?: string; // Allowed file types
}

const FileUpload: React.FC<FileUploadProps> = ({ onFileUploaded, accept = '.pdf,.txt' }) => {
  const props: UploadProps = {
    name: 'file',
    accept: accept,
    beforeUpload: (file) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        if (e.target?.result && onFileUploaded) {
          onFileUploaded(e.target.result as string);
        }
      };
      reader.readAsText(file);
      // Prevent default upload behavior
      return false;
    },
    onChange(info) {
      if (info.file.status === 'done') {
        message.success(`${info.file.name} file uploaded successfully`);
      } else if (info.file.status === 'error') {
        message.error(`${info.file.name} file upload failed.`);
      }
    },
  };

  return (
    <Upload {...props}>
      <button>
        <UploadOutlined /> Click to Upload File
      </button>
    </Upload>
  );
};

export default FileUpload;