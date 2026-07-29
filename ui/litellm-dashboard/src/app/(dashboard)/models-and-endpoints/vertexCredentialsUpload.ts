import type { FormInstance, UploadProps } from "antd";
import NotificationsManager from "@/components/molecules/notifications_manager";

export function vertexCredentialsUploadProps(form: FormInstance): UploadProps {
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
