import { useMutation } from "@tanstack/react-query";
import { loginCall, LoginRequest } from "@/components/networking";

export const useLogin = () => {
  return useMutation({
    mutationFn: async ({ username, password }: LoginRequest) => {
      const result = await loginCall(username, password);
      return result;
    },
    onSuccess: (data) => {
      // Redirect to UI on successful login
      if (data.success && data.redirectUrl) {
        window.location.href = data.redirectUrl;
      }
    },
  });
};
