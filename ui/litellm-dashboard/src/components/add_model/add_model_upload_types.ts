/**
 * Minimal upload type compatible with the antd `UploadProps` shape used
 * by the out-of-scope parent `ModelsAndEndpointsView.tsx` and by
 * `model_add/AddCredentialModal.tsx` / `model_add/EditCredentialModal.tsx`,
 * which still import the antd `UploadProps` type directly. We only depend
 * on the fields that `provider_specific_fields.tsx` actually reads
 * (`onChange` with a `file` payload exposing `name` / `status` / `type`),
 * so the `onChange` signature is typed with an `any` info payload to
 * accept both the antd `UploadChangeParam<UploadFile<any>>` shape and our
 * minimal shape — this can be tightened once all callers migrate off antd.
 */
export interface MinimalUploadFileInfo {
  name: string;
  status?: string;
  type?: string;
}

export interface MinimalUploadChangeParam {
  file: MinimalUploadFileInfo;
}

export interface UploadProps {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  onChange?: (info: any) => void;
}
