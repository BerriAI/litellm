import { basename } from "path";

const NEXT_RESERVED = new Set([
  "page",
  "layout",
  "route",
  "template",
  "default",
  "loading",
  "error",
  "global-error",
  "not-found",
  "middleware",
  "instrumentation",
  "sitemap",
  "robots",
  "manifest",
  "icon",
  "apple-icon",
  "favicon",
  "opengraph-image",
  "twitter-image",
]);

const PASCAL_CASE = /^[A-Z][A-Za-z0-9]*$/;

const rule = {
  meta: {
    type: "suggestion",
    docs: {
      description: "Require PascalCase filenames for .tsx modules; exempt Next.js reserved files and test/spec files.",
    },
    schema: [],
    messages: {
      notPascalCase: "Filename '{{name}}' should be PascalCase (e.g. '{{suggestion}}.tsx').",
    },
  },
  create(context) {
    const filename = context.filename;
    const stem = basename(filename).replace(/\.tsx$/, "");
    const [head, ...rest] = stem.split(".");
    if (rest.includes("test") || rest.includes("spec")) return {};
    if (NEXT_RESERVED.has(head)) return {};
    if (PASCAL_CASE.test(head)) return {};
    const pascalHead = head
      .split(/[-_]/)
      .filter(Boolean)
      .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
      .join("");
    const suggestion = [pascalHead, ...rest].join(".");
    return {
      Program(node) {
        context.report({ node, messageId: "notPascalCase", data: { name: `${stem}.tsx`, suggestion } });
      },
    };
  },
};

export default rule;
