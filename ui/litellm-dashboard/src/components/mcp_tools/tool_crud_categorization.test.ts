import { describe, it, expect } from "vitest";
import {
  categorizeTool,
  categorizeTools,
  groupToolsByCategory,
  type CategorizedTool,
} from "./tool_crud_categorization";

describe("tool_crud_categorization", () => {
  describe("categorizeTool", () => {
    it("should categorize Create operations", () => {
      expect(categorizeTool("createUser")).toBe("Create");
      expect(categorizeTool("addComment")).toBe("Create");
      expect(categorizeTool("insertRecord")).toBe("Create");
      expect(categorizeTool("postMessage")).toBe("Create");
      expect(categorizeTool("registerAccount")).toBe("Create");
    });

    it("should categorize Read operations", () => {
      expect(categorizeTool("getUser")).toBe("Read");
      expect(categorizeTool("fetchData")).toBe("Read");
      expect(categorizeTool("listItems")).toBe("Read");
      expect(categorizeTool("searchRecords")).toBe("Read");
      expect(categorizeTool("findUser")).toBe("Read");
      expect(categorizeTool("queryDatabase")).toBe("Read");
      expect(categorizeTool("atlassianUserInfo")).toBe("Read");
      expect(categorizeTool("getConfluencePage")).toBe("Read");
    });

    it("should categorize Update operations", () => {
      expect(categorizeTool("updateUser")).toBe("Update");
      expect(categorizeTool("editProfile")).toBe("Update");
      expect(categorizeTool("modifySettings")).toBe("Update");
      expect(categorizeTool("patchResource")).toBe("Update");
      expect(categorizeTool("changePassword")).toBe("Update");
    });

    it("should categorize Delete operations", () => {
      expect(categorizeTool("deleteUser")).toBe("Delete");
      expect(categorizeTool("removeItem")).toBe("Delete");
      expect(categorizeTool("dropTable")).toBe("Delete");
      expect(categorizeTool("clearCache")).toBe("Delete");
      expect(categorizeTool("destroySession")).toBe("Delete");
    });

    it("should categorize as Other when no match", () => {
      expect(categorizeTool("calculateSum")).toBe("Other");
      expect(categorizeTool("processPayment")).toBe("Other");
      expect(categorizeTool("validateInput")).toBe("Other");
    });

    it("should use description for categorization", () => {
      expect(categorizeTool("userTool", "Gets user information")).toBe("Read");
      expect(categorizeTool("dataTool", "Creates new records")).toBe("Create");
      expect(categorizeTool("toolName", "Updates existing data")).toBe("Update");
      expect(categorizeTool("actionTool", "Deletes old files")).toBe("Delete");
    });

    it("should be case insensitive", () => {
      expect(categorizeTool("GETUSER")).toBe("Read");
      expect(categorizeTool("CreateRecord")).toBe("Create");
      expect(categorizeTool("UpdateItem")).toBe("Update");
      expect(categorizeTool("DeleteFile")).toBe("Delete");
    });
  });

  describe("categorizeTools", () => {
    it("should categorize multiple tools", () => {
      const tools = [
        { name: "getUser", description: "Get user info" },
        { name: "createPost", description: "Create a new post" },
        { name: "updateProfile" },
        { name: "deleteComment" },
      ];

      const categorized = categorizeTools(tools);

      expect(categorized).toHaveLength(4);
      expect(categorized[0].category).toBe("Read");
      expect(categorized[1].category).toBe("Create");
      expect(categorized[2].category).toBe("Update");
      expect(categorized[3].category).toBe("Delete");
    });

    it("should handle empty array", () => {
      const categorized = categorizeTools([]);
      expect(categorized).toHaveLength(0);
    });
  });

  describe("groupToolsByCategory", () => {
    it("should group tools by CRUD category", () => {
      const tools: CategorizedTool[] = [
        { name: "getUser", category: "Read" },
        { name: "createPost", category: "Create" },
        { name: "updateProfile", category: "Update" },
        { name: "deleteComment", category: "Delete" },
        { name: "fetchData", category: "Read" },
        { name: "addItem", category: "Create" },
        { name: "processTask", category: "Other" },
      ];

      const grouped = groupToolsByCategory(tools);

      expect(grouped.Read).toHaveLength(2);
      expect(grouped.Create).toHaveLength(2);
      expect(grouped.Update).toHaveLength(1);
      expect(grouped.Delete).toHaveLength(1);
      expect(grouped.Other).toHaveLength(1);
    });

    it("should handle empty categories", () => {
      const tools: CategorizedTool[] = [
        { name: "getUser", category: "Read" },
      ];

      const grouped = groupToolsByCategory(tools);

      expect(grouped.Read).toHaveLength(1);
      expect(grouped.Create).toHaveLength(0);
      expect(grouped.Update).toHaveLength(0);
      expect(grouped.Delete).toHaveLength(0);
      expect(grouped.Other).toHaveLength(0);
    });

    it("should handle empty array", () => {
      const grouped = groupToolsByCategory([]);

      expect(grouped.Read).toHaveLength(0);
      expect(grouped.Create).toHaveLength(0);
      expect(grouped.Update).toHaveLength(0);
      expect(grouped.Delete).toHaveLength(0);
      expect(grouped.Other).toHaveLength(0);
    });
  });

  describe("real-world MCP tool examples", () => {
    it("should categorize Confluence/Atlassian tools correctly", () => {
      expect(categorizeTool("atlassianUserInfo")).toBe("Read");
      expect(categorizeTool("getAccessibleAtlassianResources")).toBe("Read");
      expect(categorizeTool("getConfluencePage")).toBe("Read");
      expect(categorizeTool("searchConfluenceUsingCql")).toBe("Read");
      expect(categorizeTool("getConfluenceSpaces")).toBe("Read");
      expect(categorizeTool("getPagesInConfluenceSpace")).toBe("Read");
      expect(categorizeTool("getConfluencePageFooterComments")).toBe("Read");
    });
  });
});
