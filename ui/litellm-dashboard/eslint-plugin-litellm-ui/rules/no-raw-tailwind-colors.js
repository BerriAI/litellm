/**
 * ESLint rule: forbid raw Tailwind color classes (e.g. `bg-slate-500`,
 * `text-blue-600`) in JSX className strings and `cn/clsx/twMerge` literal args.
 * Require semantic shadcn tokens (`bg-primary`, `text-foreground`,
 * `border-border`, etc.).
 *
 * Heuristic: a token matching
 *   /^(bg|text|border|ring|outline|fill|stroke|placeholder|divide|from|via|to)-
 *     (slate|gray|zinc|neutral|stone|red|orange|amber|yellow|lime|green|emerald|
 *      teal|cyan|sky|blue|indigo|violet|purple|fuchsia|pink|rose)-\d{2,3}$/
 * anywhere inside a `className=...` string literal, `className={`...`}` template
 * quasi, or a string argument to `cn(...)`, `clsx(...)`, or `twMerge(...)` is
 * flagged.
 *
 * Allowed classes (not flagged): semantic tokens such as `bg-background`,
 * `bg-primary`, `text-foreground`, `border-border`, `bg-muted`,
 * `text-muted-foreground`, etc. — these have no raw palette name in them.
 */
"use strict";

const RAW_COLOR_RE =
  /\b(bg|text|border|ring|outline|fill|stroke|placeholder|divide|from|via|to)-(slate|gray|zinc|neutral|stone|red|orange|amber|yellow|lime|green|emerald|teal|cyan|sky|blue|indigo|violet|purple|fuchsia|pink|rose)-\d{2,3}\b/;

function report(context, node, value) {
  const m = value.match(RAW_COLOR_RE);
  if (m) {
    context.report({
      node,
      message: `Raw Tailwind color class "${m[0]}" is banned. Use a semantic token (bg-primary, text-foreground, bg-muted, border-border, ...).`,
    });
  }
}

module.exports = {
  meta: {
    type: "problem",
    docs: {
      description:
        "Forbid raw Tailwind color classes; require shadcn semantic tokens (bg-primary, text-foreground, ...).",
    },
    schema: [],
  },
  create(context) {
    return {
      JSXAttribute(node) {
        if (node.name && node.name.name !== "className") return;
        if (!node.value) return;

        if (node.value.type === "Literal" && typeof node.value.value === "string") {
          report(context, node.value, node.value.value);
          return;
        }

        if (node.value.type === "JSXExpressionContainer") {
          const expr = node.value.expression;
          if (!expr) return;

          if (expr.type === "Literal" && typeof expr.value === "string") {
            report(context, expr, expr.value);
            return;
          }

          if (expr.type === "TemplateLiteral") {
            for (const q of expr.quasis) {
              report(context, q, q.value.raw);
            }
          }
        }
      },

      CallExpression(node) {
        const callee = node.callee;
        if (!callee || callee.type !== "Identifier") return;
        if (!["cn", "clsx", "twMerge", "cva", "cx"].includes(callee.name)) return;

        for (const arg of node.arguments) {
          if (arg.type === "Literal" && typeof arg.value === "string") {
            report(context, arg, arg.value);
          } else if (arg.type === "TemplateLiteral") {
            for (const q of arg.quasis) {
              report(context, q, q.value.raw);
            }
          }
        }
      },
    };
  },
};
