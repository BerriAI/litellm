import { KeyResponse } from "./key_list";

export const transformKeyInfo = (apiResponse: any): KeyResponse => {
  const { key, info } = apiResponse;

  // Simply combine the key with all info fields
  return {
    token: key,
    ...info,
  } as KeyResponse;
};
