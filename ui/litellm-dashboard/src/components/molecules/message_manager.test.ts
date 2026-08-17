import { beforeEach, describe, expect, it, vi } from "vitest";

const sonner = vi.hoisted(() => ({
  success: vi.fn(),
  info: vi.fn(),
  warning: vi.fn(),
  error: vi.fn(),
  dismiss: vi.fn(),
}));

vi.mock("sonner", () => ({ toast: sonner }));
vi.mock("@/components/molecules/notifications_manager", () =>
  vi.importActual<typeof import("./notifications_manager")>("./notifications_manager"),
);

import MessageManager from "./message_manager";
import NotificationManager from "./notifications_manager";

describe("legacy toast facades", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("MessageManager converts antd-era seconds into milliseconds", () => {
    MessageManager.error("failed!", 5);
    expect(sonner.error).toHaveBeenCalledWith("failed!", { description: undefined, duration: 5000 });
  });

  it("MessageManager falls back to the kind default when no duration is given", () => {
    MessageManager.success("done!");
    expect(sonner.success).toHaveBeenCalledWith("done!", { description: undefined, duration: 4000 });
  });

  it("MessageManager.destroy and NotificationManager.clear both dismiss", () => {
    MessageManager.destroy();
    NotificationManager.clear();
    expect(sonner.dismiss).toHaveBeenCalledTimes(2);
  });

  it("NotificationManager.fromBackend routes through toast.fromError", () => {
    NotificationManager.fromBackend({ message: "Team not found", type: "not_found_error", code: "404" });
    expect(sonner.error).toHaveBeenCalledWith("Not Found", { description: "Team not found", duration: 6000 });
  });
});
