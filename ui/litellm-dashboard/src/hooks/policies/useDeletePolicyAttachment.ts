import { useMutation } from "@tanstack/react-query";
import { deletePolicyAttachmentCall } from "@/components/networking";
import { toast } from "@/lib/toast";

interface UseDeletePolicyAttachmentProps {
  accessToken: string | null;
  onSuccess?: () => void;
  onError?: (error: any) => void;
}

export const useDeletePolicyAttachment = ({ accessToken, onSuccess, onError }: UseDeletePolicyAttachmentProps) => {
  return useMutation({
    mutationFn: async (attachmentId: string) => {
      if (!accessToken) {
        throw new Error("Access token is required");
      }
      return deletePolicyAttachmentCall(accessToken, attachmentId);
    },
    onSuccess: () => {
      toast.success("Attachment deleted successfully");
      if (onSuccess) {
        onSuccess();
      }
    },
    onError: (error) => {
      console.error("Error deleting attachment:", error);
      toast.error("Failed to delete attachment");
      if (onError) {
        onError(error);
      }
    },
  });
};
