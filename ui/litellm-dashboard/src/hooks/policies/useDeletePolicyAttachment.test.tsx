import React from "react";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useDeletePolicyAttachment } from "./useDeletePolicyAttachment";
import { deletePolicyAttachmentCall } from "@/components/networking";
import MessageManager from "@/components/molecules/message_manager";
import { vi, describe, beforeEach, it, expect } from "vitest";

// Mock dependencies
vi.mock("@/components/networking", () => ({
  deletePolicyAttachmentCall: vi.fn(),
}));

vi.mock("@/components/molecules/message_manager", () => ({
  default: {
    success: vi.fn(),
    error: vi.fn(),
  },
}));

describe("useDeletePolicyAttachment", () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient();
    vi.clearAllMocks();
  });

  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );

  it("should successfully delete a policy attachment and call onSuccess", async () => {
    const mockOnSuccess = vi.fn();
    (deletePolicyAttachmentCall as any).mockResolvedValue({});

    const { result } = renderHook(
      () =>
        useDeletePolicyAttachment({
          accessToken: "test-token",
          onSuccess: mockOnSuccess,
        }),
      { wrapper }
    );

    result.current.mutate("attachment-1");

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(deletePolicyAttachmentCall).toHaveBeenCalledWith("test-token", "attachment-1");
    expect(MessageManager.success).toHaveBeenCalledWith("Attachment deleted successfully");
    expect(mockOnSuccess).toHaveBeenCalled();
  });

  it("should handle error when deleting policy attachment", async () => {
    const mockOnError = vi.fn();
    const error = new Error("Delete failed");
    (deletePolicyAttachmentCall as any).mockRejectedValue(error);

    const { result } = renderHook(
      () =>
        useDeletePolicyAttachment({
          accessToken: "test-token",
          onError: mockOnError,
        }),
      { wrapper }
    );

    result.current.mutate("attachment-1");

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(deletePolicyAttachmentCall).toHaveBeenCalledWith("test-token", "attachment-1");
    expect(MessageManager.error).toHaveBeenCalledWith("Failed to delete attachment");
    expect(mockOnError).toHaveBeenCalledWith(error);
  });
});
