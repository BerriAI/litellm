import { useMutation, useQuery } from "@tanstack/react-query";
import { forgotPasswordCall, resetPasswordCall, validateResetTokenCall } from "@/components/networking";
import { createQueryKeys } from "../common/queryKeysFactory";

const passwordResetKeys = createQueryKeys("passwordReset");

export const useForgotPassword = () => {
  return useMutation({
    mutationFn: async (email: string) => forgotPasswordCall(email),
  });
};

export interface ResetTokenValidation {
  user_email: string;
}

export const useValidateResetToken = (token: string | null) => {
  return useQuery<ResetTokenValidation>({
    queryKey: passwordResetKeys.detail(token ?? ""),
    queryFn: async () => {
      if (!token) throw new Error("token is required");
      return validateResetTokenCall(token);
    },
    enabled: Boolean(token),
    retry: false,
  });
};

export interface ResetPasswordParams {
  token: string;
  newPassword: string;
}

export const useResetPassword = () => {
  return useMutation({
    mutationFn: async ({ token, newPassword }: ResetPasswordParams) => resetPasswordCall(token, newPassword),
  });
};
