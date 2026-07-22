import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Form } from "antd";
import React from "react";
import { describe, expect, it, vi } from "vitest";
import MetadataKeyValueFields, {
  MetadataPair,
  metadataObjectToPairs,
  metadataPairsToObject,
} from "./MetadataKeyValueFields";

describe("metadataObjectToPairs", () => {
  it("returns an empty list for null or undefined metadata", () => {
    expect(metadataObjectToPairs(null)).toEqual([]);
    expect(metadataObjectToPairs(undefined)).toEqual([]);
  });

  it("keeps plain string values as-is", () => {
    expect(metadataObjectToPairs({ department: "research" })).toEqual([{ key: "department", value: "research" }]);
  });

  it("serializes non-string values as JSON", () => {
    expect(
      metadataObjectToPairs({
        tier: 3,
        beta: true,
        config: { region: "us" },
        tags: ["a", "b"],
        empty: null,
      }),
    ).toEqual([
      { key: "tier", value: "3" },
      { key: "beta", value: "true" },
      { key: "config", value: '{"region":"us"}' },
      { key: "tags", value: '["a","b"]' },
      { key: "empty", value: "null" },
    ]);
  });

  it("quotes string values that would otherwise parse as JSON, so types round-trip", () => {
    expect(metadataObjectToPairs({ code: "42", flag: "true" })).toEqual([
      { key: "code", value: '"42"' },
      { key: "flag", value: '"true"' },
    ]);
  });

  it("filters out excluded keys", () => {
    expect(
      metadataObjectToPairs({ department: "research", logging: [{ callback_name: "langfuse" }] }, new Set(["logging"])),
    ).toEqual([{ key: "department", value: "research" }]);
  });
});

describe("metadataPairsToObject", () => {
  it("returns an empty object for undefined pairs", () => {
    expect(metadataPairsToObject(undefined)).toEqual({});
  });

  it("keeps plain text values as strings", () => {
    expect(metadataPairsToObject([{ key: "department", value: "research" }])).toEqual({ department: "research" });
  });

  it("parses JSON values into their typed form", () => {
    expect(
      metadataPairsToObject([
        { key: "tier", value: "3" },
        { key: "beta", value: "true" },
        { key: "config", value: '{"region":"us"}' },
        { key: "code", value: '"42"' },
      ]),
    ).toEqual({ tier: 3, beta: true, config: { region: "us" }, code: "42" });
  });

  it("skips rows without a key and defaults a missing value to an empty string", () => {
    expect(metadataPairsToObject([{ key: "", value: "orphan" }, undefined, { key: "kept" }])).toEqual({ kept: "" });
  });

  it("round-trips a mixed-type metadata object losslessly", () => {
    const metadata = {
      department: "research",
      code: "42",
      tier: 3,
      beta: true,
      config: { region: "us", replicas: 2 },
    };
    expect(metadataPairsToObject(metadataObjectToPairs(metadata))).toEqual(metadata);
  });
});

interface HarnessProps {
  onFinish: (values: { metadata?: MetadataPair[] }) => void;
  initialMetadata?: MetadataPair[];
}

const Harness: React.FC<HarnessProps> = ({ onFinish, initialMetadata }) => {
  const [form] = Form.useForm();
  return (
    <Form form={form} onFinish={onFinish} initialValues={{ metadata: initialMetadata }}>
      <MetadataKeyValueFields form={form} />
      <button type="submit">Save</button>
    </Form>
  );
};

describe("MetadataKeyValueFields", () => {
  it("renders one row per existing pair", () => {
    render(
      <Harness
        onFinish={vi.fn()}
        initialMetadata={[
          { key: "department", value: "research" },
          { key: "tier", value: "3" },
        ]}
      />,
    );

    const keyInputs = screen.getAllByPlaceholderText("Key");
    const valueInputs = screen.getAllByPlaceholderText("Value");
    expect(keyInputs.map((input) => (input as HTMLInputElement).value)).toEqual(["department", "tier"]);
    expect(valueInputs.map((input) => (input as HTMLInputElement).value)).toEqual(["research", "3"]);
  });

  it("adds a row and submits the entered pair", async () => {
    const user = userEvent.setup();
    const onFinish = vi.fn();
    render(<Harness onFinish={onFinish} />);

    await user.click(screen.getByRole("button", { name: /add key-value pair/i }));
    await user.type(screen.getByPlaceholderText("Key"), "cost_center");
    await user.type(screen.getByPlaceholderText("Value"), "eng-1");
    await user.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => {
      expect(onFinish).toHaveBeenCalledWith({ metadata: [{ key: "cost_center", value: "eng-1" }] });
    });
  });

  it("removes a row when its remove icon is clicked", async () => {
    const user = userEvent.setup();
    const onFinish = vi.fn();
    render(
      <Harness
        onFinish={onFinish}
        initialMetadata={[
          { key: "department", value: "research" },
          { key: "tier", value: "3" },
        ]}
      />,
    );

    await user.click(screen.getAllByLabelText("Remove key-value pair")[0]);
    await user.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => {
      expect(onFinish).toHaveBeenCalledWith({ metadata: [{ key: "tier", value: "3" }] });
    });
  });

  it("blocks submission on duplicate keys", async () => {
    const user = userEvent.setup();
    const onFinish = vi.fn();
    render(
      <Harness
        onFinish={onFinish}
        initialMetadata={[
          { key: "department", value: "research" },
          { key: "department", value: "sales" },
        ]}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => {
      expect(screen.getAllByText("Duplicate key").length).toBeGreaterThan(0);
    });
    expect(onFinish).not.toHaveBeenCalled();
  });

  it("blocks submission when a row is missing its key", async () => {
    const user = userEvent.setup();
    const onFinish = vi.fn();
    render(<Harness onFinish={onFinish} />);

    await user.click(screen.getByRole("button", { name: /add key-value pair/i }));
    await user.type(screen.getByPlaceholderText("Value"), "orphan");
    await user.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => {
      expect(screen.getByText("Missing key")).toBeInTheDocument();
    });
    expect(onFinish).not.toHaveBeenCalled();
  });
});
