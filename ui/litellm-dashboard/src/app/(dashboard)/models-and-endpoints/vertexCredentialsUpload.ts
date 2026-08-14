import type { ComponentProps } from "react";

import NotificationsManager from "@/components/molecules/notifications_manager";
import type CredentialsPanel from "@/components/model_add/CredentialsPanel";

interface VertexCredentialsForm {
  setFieldsValue: (values: { vertex_credentials: string }) => void;
}

type UploadProps = ComponentProps<typeof CredentialsPanel>["uploadProps"];

export function vertexCredentialsUploadProps(form: VertexCredentialsForm): UploadProps {
  return {
    name: "file",
    accept: ".json",
    pastable: false,
    beforeUpload: (file) => {
      if (file.type === "application/json") {
        const reader = new FileReader();
        reader.onload = (event) => {
          if (event.target) {
            form.setFieldsValue({ vertex_credentials: event.target.result as string });
          }
        };
        reader.readAsText(file);
      }
      return false;
    },
    onChange(info) {
      if (info.file.status === "done") {
        NotificationsManager.success(`${info.file.name} file uploaded successfully`);
      } else if (info.file.status === "error") {
        NotificationsManager.fromBackend(`${info.file.name} file upload failed.`);
      }
    },
  };
}
