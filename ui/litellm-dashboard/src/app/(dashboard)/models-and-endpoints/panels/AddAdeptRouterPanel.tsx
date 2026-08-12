"use client";

import { Form } from "antd";
import { useQueryClient } from "@tanstack/react-query";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import AddAdeptRouterTab from "@/components/add_model/AddAdeptRouterTab";

export default function AddAdeptRouterPanel() {
  const { accessToken, userRole } = useAuthorized();
  const [form] = Form.useForm();
  const queryClient = useQueryClient();

  return (
    <AddAdeptRouterTab
      form={form}
      onSuccess={() => queryClient.invalidateQueries({ queryKey: ["models", "list"] })}
      accessToken={accessToken ?? ""}
      userRole={userRole ?? ""}
    />
  );
}
