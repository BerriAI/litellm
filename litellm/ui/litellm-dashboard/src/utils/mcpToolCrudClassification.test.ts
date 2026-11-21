import { describe, it, expect } from "vitest";
import { classifyToolOp, groupToolsByCrud } from "./mcpToolCrudClassification";

describe("classifyToolOp", () => {
  it("should classify read operations by name", () => {
    expect(classifyToolOp("get-users")).toBe("read");
    expect(classifyToolOp("list-items")).toBe("read");
    expect(classifyToolOp("search documents")).toBe("read");
  });

  it("should classify delete operations by name", () => {
    expect(classifyToolOp("delete-user")).toBe("delete");
    expect(classifyToolOp("remove-item")).toBe("delete");
    expect(classifyToolOp("purge-cache")).toBe("delete");
  });

  it("should classify create operations by name", () => {
    expect(classifyToolOp("create-user")).toBe("create");
    expect(classifyToolOp("add-item")).toBe("create");
    expect(classifyToolOp("upload-file")).toBe("create");
  });

  it("should classify update operations by name", () => {
    expect(classifyToolOp("update-settings")).toBe("update");
    expect(classifyToolOp("edit-profile")).toBe("update");
    expect(classifyToolOp("rename-file")).toBe("update");
  });

  it("should prioritize read over delete for names like get-removed-entries", () => {
    expect(classifyToolOp("get-removed-entries")).toBe("read");
    expect(classifyToolOp("list-deleted-items")).toBe("read");
  });

  it("should fall back to description when name is unrecognised", () => {
    expect(classifyToolOp("mytool", "This will delete the record")).toBe("delete");
    expect(classifyToolOp("mytool", "fetch data from the API")).toBe("read");
  });

  it("should return unknown when neither name nor description match", () => {
    expect(classifyToolOp("my_tool")).toBe("unknown");
    expect(classifyToolOp("my_tool", "does something")).toBe("unknown");
  });
});

describe("groupToolsByCrud", () => {
  it("should group tools into their CRUD categories", () => {
    const tools = [
      { name: "get-user", description: "" },
      { name: "create-item", description: "" },
      { name: "delete-record", description: "" },
      { name: "update-settings", description: "" },
      { name: "mysteryop", description: "" },
    ];

    const groups = groupToolsByCrud(tools);

    expect(groups.read).toHaveLength(1);
    expect(groups.create).toHaveLength(1);
    expect(groups.delete).toHaveLength(1);
    expect(groups.update).toHaveLength(1);
    expect(groups.unknown).toHaveLength(1);
  });
});
