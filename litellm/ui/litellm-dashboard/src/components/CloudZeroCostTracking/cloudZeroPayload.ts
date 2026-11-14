export interface CloudZeroFormValues {
  api_key: string;
  connection_id: string;
  timezone: string;
}

export interface CloudZeroPayload {
  connection_id: string;
  timezone: string;
  api_key?: string;
}

export const EMPTY_CLOUDZERO_FORM_VALUES: CloudZeroFormValues = {
  api_key: "",
  connection_id: "",
  timezone: "",
};

export const buildCloudZeroPayload = (values: CloudZeroFormValues): CloudZeroPayload => ({
  connection_id: values.connection_id,
  timezone: values.timezone || "UTC",
  ...(values.api_key && { api_key: values.api_key }),
});
