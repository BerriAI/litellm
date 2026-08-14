import { waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import NotificationsManager from "@/components/molecules/notifications_manager";

import { vertexCredentialsUploadProps } from "./vertexCredentialsUpload";

vi.mock("@/components/molecules/notifications_manager", () => ({
  default: { success: vi.fn(), fromBackend: vi.fn() },
}));

const makeForm = () => ({ setFieldsValue: vi.fn() });

describe("vertexCredentialsUploadProps", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("reads a JSON credential file into the vertex_credentials field without uploading it", async () => {
    const form = makeForm();
    const props = vertexCredentialsUploadProps(form as never);
    const file = new File(['{"project_id":"example"}'], "vertex.json", { type: "application/json" });

    expect(props.beforeUpload?.(file as never, [file] as never)).toBe(false);

    await waitFor(() => {
      expect(form.setFieldsValue).toHaveBeenCalledWith({ vertex_credentials: '{"project_id":"example"}' });
    });
  });

  it("ignores non-JSON files", async () => {
    const form = makeForm();
    const props = vertexCredentialsUploadProps(form as never);
    const file = new File(["not json"], "vertex.txt", { type: "text/plain" });

    expect(props.beforeUpload?.(file as never, [file] as never)).toBe(false);

    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(form.setFieldsValue).not.toHaveBeenCalled();
  });

  it("reports completed and failed upload states", () => {
    const props = vertexCredentialsUploadProps(makeForm() as never);

    props.onChange?.({ file: { name: "vertex.json", status: "done" } } as never);
    props.onChange?.({ file: { name: "vertex.json", status: "error" } } as never);

    expect(NotificationsManager.success).toHaveBeenCalledWith("vertex.json file uploaded successfully");
    expect(NotificationsManager.fromBackend).toHaveBeenCalledWith("vertex.json file upload failed.");
  });
});
