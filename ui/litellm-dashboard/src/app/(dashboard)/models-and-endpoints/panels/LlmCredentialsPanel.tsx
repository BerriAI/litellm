"use client";

import { useForm } from "react-hook-form";
import CredentialsPanel from "@/components/model_add/CredentialsPanel";
import type { MountedFormValues } from "@/components/common_components/MountedFormField";
import { vertexCredentialsUploadProps } from "@/app/(dashboard)/models-and-endpoints/vertexCredentialsUpload";

export default function LlmCredentialsPanel() {
  const form = useForm<MountedFormValues>();
  return (
    <CredentialsPanel
      uploadProps={vertexCredentialsUploadProps({
        setFieldsValue: (values) => form.setValue("vertex_credentials", values.vertex_credentials),
      })}
    />
  );
}
