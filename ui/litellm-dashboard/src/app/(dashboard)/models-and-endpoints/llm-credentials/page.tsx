"use client";

import { Form } from "antd";
import CredentialsPanel from "@/components/model_add/CredentialsPanel";
import { vertexCredentialsUploadProps } from "@/app/(dashboard)/models-and-endpoints/vertexCredentialsUpload";

export default function LlmCredentialsPage() {
  const [form] = Form.useForm();
  return <CredentialsPanel uploadProps={vertexCredentialsUploadProps(form)} />;
}
