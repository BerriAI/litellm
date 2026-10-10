import { apiClient } from "@/components/networking";
import type { components } from "@/lib/http/schema";
import type { EndUser } from "./useCustomers";

type Schemas = components["schemas"];

export type CreateCustomerRequest = Partial<Schemas["NewCustomerRequest"]> & { user_id: string };
export type UpdateCustomerRequest = Partial<Schemas["UpdateCustomerRequest"]> & { user_id: string };
export type CustomerBlockedStateInput = { userId: string; blocked: boolean };

export const createCustomer = (accessToken: string, body: CreateCustomerRequest): Promise<EndUser> =>
  apiClient.post<EndUser>("/customer/new", { accessToken, body });

export const updateCustomer = (accessToken: string, body: UpdateCustomerRequest): Promise<EndUser> =>
  apiClient.post<EndUser>("/customer/update", { accessToken, body });

export const deleteCustomers = (accessToken: string, userIds: string[]): Promise<Schemas["DeleteCustomersResponse"]> =>
  apiClient.post<Schemas["DeleteCustomersResponse"]>("/customer/delete", {
    accessToken,
    body: { user_ids: userIds },
  });

export const setCustomerBlockedState = (
  accessToken: string,
  { userId, blocked }: CustomerBlockedStateInput,
): Promise<EndUser> => updateCustomer(accessToken, { user_id: userId, blocked });
