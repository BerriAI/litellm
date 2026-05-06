import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, act, fireEvent } from "@testing-library/react";
import { Form } from "antd";
import OAuthFormFields from "./OAuthFormFields";

// ── helpers ──────────────────────────────────────────────────────────────────

/** Minimal Ant Form wrapper so Form.Item registers correctly. */
const WithForm: React.FC<{ children: React.ReactNode; onFinish?: (values: any) => void }> = ({
  children,
  onFinish,
}) => {
  const [form] = Form.useForm();
  return (
    <Form form={form} onFinish={onFinish}>
      {children}
      <button type="submit">Submit</button>
    </Form>
  );
};

// ── tests ─────────────────────────────────────────────────────────────────────

describe("OAuthFormFields", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  // ── visibility by flow type ─────────────────────────────────────────────────

  describe("interactive mode (isM2M=false)", () => {
    it("renders Token Validation Rules field", () => {
      render(
        <WithForm>
          <OAuthFormFields isM2M={false} />
        </WithForm>,
      );
      expect(screen.getByText("Token Validation Rules (optional)")).toBeInTheDocument();
    });

    it("renders Token Storage TTL field", () => {
      render(
        <WithForm>
          <OAuthFormFields isM2M={false} />
        </WithForm>,
      );
      expect(screen.getByText("Token Storage TTL (seconds, optional)")).toBeInTheDocument();
    });

    it("renders standard interactive fields alongside the new fields", () => {
      render(
        <WithForm>
          <OAuthFormFields isM2M={false} />
        </WithForm>,
      );
      expect(screen.getByText("Authorization URL (optional)")).toBeInTheDocument();
      expect(screen.getByText("Registration URL (optional)")).toBeInTheDocument();
      expect(screen.getByText("Token Validation Rules (optional)")).toBeInTheDocument();
      expect(screen.getByText("Token Storage TTL (seconds, optional)")).toBeInTheDocument();
    });
  });

  describe("M2M mode (isM2M=true)", () => {
    it("does NOT render Token Validation Rules field", () => {
      render(
        <WithForm>
          <OAuthFormFields isM2M={true} />
        </WithForm>,
      );
      expect(screen.queryByText("Token Validation Rules (optional)")).not.toBeInTheDocument();
    });

    it("does NOT render Token Storage TTL field", () => {
      render(
        <WithForm>
          <OAuthFormFields isM2M={true} />
        </WithForm>,
      );
      expect(screen.queryByText("Token Storage TTL (seconds, optional)")).not.toBeInTheDocument();
    });

    it("still renders M2M-specific fields", () => {
      render(
        <WithForm>
          <OAuthFormFields isM2M={true} />
        </WithForm>,
      );
      expect(screen.getByText("Client ID")).toBeInTheDocument();
      expect(screen.getByText("Token URL")).toBeInTheDocument();
    });
  });

  // ── token_validation_json inline JSON validator ──────────────────────────────

  describe("token_validation_json validation", () => {
    it("accepts empty value without error", async () => {
      const onFinish = vi.fn();
      render(
        <WithForm onFinish={onFinish}>
          <OAuthFormFields isM2M={false} />
        </WithForm>,
      );

      // Leave the textarea empty and submit
      const submitBtn = screen.getByRole("button", { name: "Submit" });
      await act(async () => {
        fireEvent.click(submitBtn);
      });

      await waitFor(() => {
        expect(screen.queryByText("Must be valid JSON")).not.toBeInTheDocument();
      });
    });

    it("accepts a valid JSON object without error", async () => {
      render(
        <WithForm>
          <OAuthFormFields isM2M={false} />
        </WithForm>,
      );

      const textarea = document.getElementById("token_validation_json") as HTMLTextAreaElement;
      await act(async () => {
        fireEvent.change(textarea, { target: { value: '{"organization": "my-org"}' } });
      });

      const submitBtn = screen.getByRole("button", { name: "Submit" });
      await act(async () => {
        fireEvent.click(submitBtn);
      });

      await waitFor(() => {
        expect(screen.queryByText("Must be valid JSON")).not.toBeInTheDocument();
      });
    });

    it("shows 'Must be valid JSON' error for malformed JSON", async () => {
      render(
        <WithForm>
          <OAuthFormFields isM2M={false} />
        </WithForm>,
      );

      const textarea = document.getElementById("token_validation_json") as HTMLTextAreaElement;
      await act(async () => {
        fireEvent.change(textarea, { target: { value: "not-valid-json{" } });
      });

      const submitBtn = screen.getByRole("button", { name: "Submit" });
      await act(async () => {
        fireEvent.click(submitBtn);
      });

      await waitFor(() => {
        expect(screen.getByText("Must be valid JSON")).toBeInTheDocument();
      });
    });

    it("shows error for a plain string value (not a JSON object)", async () => {
      render(
        <WithForm>
          <OAuthFormFields isM2M={false} />
        </WithForm>,
      );

      const textarea = document.getElementById("token_validation_json") as HTMLTextAreaElement;
      await act(async () => {
        // A bare string is valid JSON but we still want to accept it; only truly
        // unparseable text should fail.  Bare "hello" is actually invalid JSON
        // (no quotes), so it should fail.
        fireEvent.change(textarea, { target: { value: "hello" } });
      });

      const submitBtn = screen.getByRole("button", { name: "Submit" });
      await act(async () => {
        fireEvent.click(submitBtn);
      });

      await waitFor(() => {
        expect(screen.getByText("Must be valid JSON")).toBeInTheDocument();
      });
    });

    it("whitespace-only value is treated as empty and passes validation", async () => {
      const onFinish = vi.fn();
      render(
        <WithForm onFinish={onFinish}>
          <OAuthFormFields isM2M={false} />
        </WithForm>,
      );

      const textarea = document.getElementById("token_validation_json") as HTMLTextAreaElement;
      await act(async () => {
        fireEvent.change(textarea, { target: { value: "   " } });
      });

      const submitBtn = screen.getByRole("button", { name: "Submit" });
      await act(async () => {
        fireEvent.click(submitBtn);
      });

      await waitFor(() => {
        expect(screen.queryByText("Must be valid JSON")).not.toBeInTheDocument();
      });
    });
  });
});
