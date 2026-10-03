import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import PublishModal from "./PublishModal";

describe("PublishModal", () => {
  it("edits and publishes a prompt name", async () => {
    const onNameChange = vi.fn();
    const onPublish = vi.fn();
    render(
      <PublishModal
        visible
        promptName="welcome"
        isSaving={false}
        onNameChange={onNameChange}
        onPublish={onPublish}
        onCancel={vi.fn()}
      />,
    );
    fireEvent.change(await screen.findByPlaceholderText("Enter prompt name"), { target: { value: "greeting" } });
    fireEvent.click(screen.getByRole("button", { name: "Publish" }));
    expect(onNameChange).toHaveBeenCalledWith("greeting");
    expect(onPublish).toHaveBeenCalledOnce();
  });
});
