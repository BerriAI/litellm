import { useMutation } from "@tanstack/react-query";
import { loginCall, LoginRequest } from "@/components/networking";

export const useLogin = () => {
  return useMutation({
    mutationFn: async ({ username, password, useV3, cliLoginId }: LoginRequest) => {
      const result = await loginCall(username, password, useV3, cliLoginId);
      return result;
    },
  });
};
