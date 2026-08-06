import { beforeEach, describe, expect, it, vi } from "vitest";
import { activeRequestsCall, cancelActiveRequestCall } from "./activeRequestsApi";
import { fetchClient } from "@/lib/http/api";

vi.mock("@/lib/http/api", () => ({
  fetchClient: { GET: vi.fn(), POST: vi.fn() },
}));

const mockedGet = vi.mocked(fetchClient.GET);
const mockedPost = vi.mocked(fetchClient.POST);

describe("activeRequestsApi", () => {
  beforeEach(() => {
    mockedGet.mockReset();
    mockedPost.mockReset();
  });

  it("should raise the proxy's reason when a cancellation is refused", async () => {
    mockedPost.mockResolvedValue({ error: { detail: "That request is no longer running" } } as never);

    await expect(cancelActiveRequestCall("reg-1")).rejects.toThrow("That request is no longer running");
  });

  it("should raise a fallback when a refused cancellation carries no reason", async () => {
    mockedPost.mockResolvedValue({ error: {} } as never);

    await expect(cancelActiveRequestCall("reg-1")).rejects.toThrow("Failed to cancel the request");
  });

  it("should return the confirmation of an accepted cancellation", async () => {
    mockedPost.mockResolvedValue({ data: { cancelled: true, detail: "Cancellation sent" } } as never);

    await expect(cancelActiveRequestCall("reg-1")).resolves.toEqual({ cancelled: true, detail: "Cancellation sent" });
    expect(mockedPost).toHaveBeenCalledWith("/global/active_requests/{registry_id}/cancel", {
      params: { path: { registry_id: "reg-1" } },
    });
  });

  it("should raise the proxy's reason when the list is refused", async () => {
    mockedGet.mockResolvedValue({
      error: { detail: "Only proxy administrators can view global active requests" },
    } as never);

    await expect(activeRequestsCall({ page: 1, page_size: 50 })).rejects.toThrow(
      "Only proxy administrators can view global active requests",
    );
  });
});
