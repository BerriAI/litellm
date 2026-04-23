/**
 * ESLint rule: forbid UI packages that are banned in phase 1 of the shadcn
 * migration.
 *
 * Banned packages:
 *   - `antd`                → use shadcn primitives in @/components/ui
 *   - `@ant-design/icons`   → use lucide-react
 *   - `@heroicons/react`    → use lucide-react
 *   - `@remixicon/react`    → use lucide-react
 *
 * Tremor (`@tremor/react`) is allowed only for chart components. The
 * exemption applies when the importing file lives in a directory named
 * `charts/` (case-insensitive) or when the file's basename contains the
 * substring `chart`. All other files must not import from `@tremor/react`.
 *
 * See the design in
 * `ui/litellm-dashboard/docs/RECIPE.md` / the phase-1 migration plan.
 */
"use strict";

const BANNED = [
  {
    from: "antd",
    message: "antd is banned in phase 1. Use shadcn primitives from @/components/ui.",
  },
  {
    from: "@ant-design/icons",
    message: "@ant-design/icons is banned in phase 1. Use lucide-react.",
  },
  {
    from: "@heroicons/react",
    message: "@heroicons/react is banned in phase 1. Use lucide-react.",
  },
  {
    from: "@remixicon/react",
    message: "@remixicon/react is banned in phase 1. Use lucide-react.",
  },
];

const TREMOR = "@tremor/react";

function isChartFile(filename) {
  if (!filename) return false;
  const lower = filename.toLowerCase();
  if (lower.includes("/charts/")) return true;
  const parts = filename.split("/");
  const basename = parts[parts.length - 1] || "";
  return basename.toLowerCase().includes("chart");
}

module.exports = {
  meta: {
    type: "problem",
    docs: {
      description:
        "Forbid UI packages banned in phase 1 of the shadcn migration (antd, antd-icons, heroicons, remixicon, and non-chart tremor imports).",
    },
    schema: [],
    messages: {
      banned: "{{message}}",
      tremorOutsideCharts:
        "@tremor/react is allowed only for charts. Move the importer under a `charts/` directory or rename it to include `chart`, or use shadcn primitives instead.",
    },
  },
  create(context) {
    const filename =
      typeof context.getFilename === "function" ? context.getFilename() : context.filename;
    const allowTremor = isChartFile(filename);
    return {
      ImportDeclaration(node) {
        const src = node.source.value;
        for (const b of BANNED) {
          if (src === b.from || src.startsWith(b.from + "/")) {
            context.report({ node, messageId: "banned", data: { message: b.message } });
            return;
          }
        }
        if (src === TREMOR && !allowTremor) {
          context.report({ node, messageId: "tremorOutsideCharts" });
        }
      },
    };
  },
};
