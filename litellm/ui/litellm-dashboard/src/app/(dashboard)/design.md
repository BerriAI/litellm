# LiteLLM dashboard UI guide

Use this guide when adding or refining a dashboard screen. Keep the existing information, permissions, navigation, and actions. Start from the shared components below, then choose the page layout that fits the task. A table, an investigation workspace, and a settings form have different space requirements.

The visual anchors are the existing LiteLLM sidebar and logo, neutral surfaces, compact controls, restrained borders, readable tables, and the shared light and dark themes. Do not introduce another font, palette, component library, or navigation shell.

## Source of truth

| Concern                               | Owner                                                                          |
| ------------------------------------- | ------------------------------------------------------------------------------ |
| Theme, semantic colors, radii, layers | [`globals.css`](../globals.css)                                                |
| Page padding and section tabs         | [`Page.tsx`](../../components/shared/Page.tsx)                                 |
| Titles, descriptions and controls     | [`PageHeader.tsx`](../../components/shared/PageHeader.tsx)                     |
| Dashboard shell                       | [`layout.tsx`](./layout.tsx)                                                   |
| Sidebar dimensions and item styles    | [`sidebarStyles.ts`](../../components/shared/sidebarStyles.ts)                 |
| Inputs, buttons, dialogs, cards, tabs | [`components/ui`](../../components/ui)                                         |
| Data tables and pagination            | [`DataTable`](../../components/shared/DataTable)                               |
| Dates and ranges                      | [`advanced_date_picker.tsx`](../../components/shared/advanced_date_picker.tsx) |
| Chat-specific composition             | [`chat/design.md`](../../components/chat/design.md)                            |

Read the actual component before using a prop. These are shadcn components backed by Base UI, with repository-specific APIs. Do not assume an API from another shadcn project applies here. Keep existing form and data libraries when restyling a screen.

## Choose a page layout

| Layout     | Use                                   | Width and scrolling                                                                                                                            |
| ---------- | ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Management | Models, keys, teams, policies, tools  | Fill the available content width. Scroll tables inside their frame.                                                                            |
| Analytics  | Usage, logs, Lens, cost reports       | Keep the existing compact title and filter composition. Charts and tables use the available width.                                             |
| Settings   | Router, admin, logging, theme         | Use the same page gutters. Keep simple forms within `max-w-4xl`, broader configuration within `max-w-6xl`. Align them to the page's left edge. |
| Workspace  | Playground, editors, inspectors       | Fill the remaining height on desktop. Let panels stack and content scroll on narrow screens.                                                   |
| Detail     | An entity's configuration or activity | Retain its existing back control and tabs. Match the parent page's gutters.                                                                    |

Do not center a narrow settings column in a laptop's remaining space and leave a large empty gutter. Do not constrain a comparison table to a reading width. The sidebar already takes space from the viewport.

## Geometry

Use the 4px spacing scale. These are defaults, not an instruction to add padding at every level.

| Relationship         | Tailwind                               | Size                            |
| -------------------- | -------------------------------------- | ------------------------------- |
| Page inset           | `p-4 sm:p-6`                           | 16px on phones, 24px from 640px |
| Major sections       | `gap-6` or `space-y-6`                 | 24px                            |
| Related field groups | `gap-4` or `space-y-4`                 | 16px                            |
| Label to input       | `gap-1.5`                              | 6px                             |
| Controls in one row  | `gap-2`                                | 8px                             |
| Title to subtitle    | `mt-1.5`                               | 6px                             |
| Subtitle to controls | `mt-4`                                 | 16px                            |
| Card content         | Existing Card spacing, or `p-4 sm:p-6` | 16–24px                         |
| Inline icon to label | Button default, usually `gap-1.5`      | 6px                             |

One owner supplies the outer page inset. A tab panel inside `Page` does not receive another `p-6` or `p-8`. A card inside a panel can have its own padding because it is a distinct group. Avoid combinations such as `w-full mx-4` that make a container wider than its parent.

Give shrinkable flex and grid children `min-w-0`. Give children that own vertical scrolling `min-h-0`. A fixed-width action should not squeeze the adjacent description to a few characters. Use a wrapping row and give the description a reasonable basis.

```tsx
<div className="flex flex-wrap items-start justify-between gap-3">
  <div className="min-w-0 basis-72 flex-1">
    <h2 className="text-lg font-semibold">Section title</h2>
    <p className="mt-1.5 text-sm text-muted-foreground">Existing description.</p>
  </div>
  <Button>Add resource</Button>
</div>
```

## Typography

Inherit the app's font. Use one clear page title, then smaller section headings. The compact title in the shell identifies the route and does not replace a management page's content heading.

| Role                                        | Default                                           |
| ------------------------------------------- | ------------------------------------------------- |
| Management or settings page title           | `PageHeaderTitle`, 24px, semibold, tight tracking |
| Compact analytics or workspace title        | Preserve the existing 16–20px inline heading      |
| Major section title                         | `text-lg font-semibold`                           |
| Card or form group title                    | `text-base font-semibold`                         |
| Field label, body, table data               | `text-sm`                                         |
| Helper text, timestamps, secondary metadata | `text-xs text-muted-foreground`                   |
| Code, IDs, configuration values             | `font-mono text-xs` or `text-sm`                  |

Descriptions use `text-sm leading-relaxed text-muted-foreground` and at most `max-w-3xl`. Keep the existing copy and translations. Do not add a subtitle when there is nothing useful to explain. Do not make a tab's content heading larger than the page heading. Use `h1`, `h2`, and `h3` according to their relationship, independently of their font size.

## Surfaces and borders

Use spacing first. Add a boundary when it identifies a table, a configuration group, an editor, or an independently interactive panel.

| Surface                          | Treatment                                                               |
| -------------------------------- | ----------------------------------------------------------------------- |
| Page                             | `bg-background`, no outer card                                          |
| Table                            | One `rounded-lg border border-border` frame, quiet `bg-muted/50` header |
| Form group or distinct panel     | Existing `Card`, with its standard spacing and radius                   |
| Inline code or supporting region | `bg-muted`, appropriate padding, readable foreground                    |
| Navigation                       | Existing `sidebar-*` tokens                                             |
| Dialog or popover                | Existing primitive and its default surface                              |

Do not surround a whole management tab with a Card and then put another bordered table inside it. Keep section headings and actions outside the table frame. Do not remove a boundary from a separate form group merely to make every page look identical. Avoid extra shadows, arbitrary border shades, and a new radius on each screen.

Use semantic colors from `globals.css`: `background`, `foreground`, `card`, `muted`, `muted-foreground`, `border`, `primary`, `destructive`, `success`, `warning`, and `info`. Preserve the theme's paired foreground colors. Use `sidebar-accent` for selected sidebar items. Generic `accent`, `secondary`, and `muted` can have the same value and may provide no contrast against one another.

Use the shared layer utilities (`z-raised`, `z-sticky`, `z-floating`, `z-popup`) according to their owners. Do not add arbitrary z-index values or restyle a portal only through an ancestor it does not render inside.

## Buttons and controls

Use `Button` from `@/components/ui/button`. The default primary action is 36px high. Use `variant="outline"` for a secondary action and `variant="ghost"` for a low-emphasis control. Use the existing destructive treatment and confirmation flow. A table's row actions may use `size="sm"` or `size="icon-sm"`; do not make a page's primary action tiny.

Keep related actions together with `flex flex-wrap items-center gap-2`. Preserve their order on mobile. Icon-only controls need an accessible name. Use existing lucide icons at 16px inline, with no decorative icon containers added to every heading.

Use the existing Input, Select, Combobox, Switch, Textarea, and form-field components. Preserve labels, descriptions, validation, defaults, and disabled states. Long selected values must not widen the page. Keep the full value available through the existing detail or tooltip treatment if a compact control truncates it.

## Tabs

Use `PageTabsList` and `PageTabsTrigger` for page sections. Their default line style has a 40px row, 24px gaps, no horizontal label padding, and an underline inside the row. The list scrolls horizontally when labels do not fit. All destinations remain available.

Do not combine `flex-wrap` with a fixed-height tab bar. Do not force every tab to shrink until its label is unreadable. Use the ordinary `TabsList` for a small segmented control such as Form/cURL or Trace/Code. Tabs inside a panel must be smaller or quieter than the page's navigation.

Keep existing URL-state behavior. Styling does not change selected-tab persistence, lazy mounting, query parameters, or permissions. Follow [the URL-state skill](../../../.agents/skills/url-state/SKILL.md) for a separate behavior change.

## Tables and empty states

Use `DataTable` for management lists and the existing shared Table primitive where the page already uses it. Keep sorting, column visibility, row selection, filtering, resizing, pagination, and keyboard behavior intact.

The table may scroll horizontally. The page must not. Keep its heading, toolbar, empty state, and pagination within the visible content width. The shared DataTable anchors empty messages to its scroll container with container-width units, so a wide set of columns cannot push the message offscreen.

Pagination wraps between its logical groups. Keep phrases such as “Rows per page” and the page count together with `whitespace-nowrap`. Preserve every column and action. Mobile styling is not permission to hide data or replace a table with a partial summary.

Verify empty, loading, error, and populated states. Include long resource names and a realistic number of columns. A screenshot of an empty table does not establish that populated rows are usable.

## Forms, dialogs and workspaces

Use one column for ordinary forms on a phone. Add columns when each field still has enough usable width. A setting with a short value and a long description can use a wrapping row. Keep the description readable and the control reachable.

Dialogs use the existing Dialog or Sheet. Give a long dialog a bounded height, for example `max-h-[calc(100dvh-2rem)] overflow-y-auto`, and an appropriate `sm:max-w-*`. Preserve the close control, labels, focus behavior and submit footer. Never set a fixed width larger than the viewport.

For a split workspace, start with a stacked layout, then add a rail when space permits.

```tsx
<div className="flex min-h-0 min-w-0 flex-1 flex-col lg:flex-row">
  <aside className="max-h-80 w-full shrink-0 overflow-auto border-b border-border lg:max-h-none lg:w-72 lg:border-r lg:border-b-0">
    {navigation}
  </aside>
  <section className="min-h-96 min-w-0 flex-1 lg:min-h-0">{editor}</section>
</div>
```

When a workspace can be embedded inside another panel, use a named container query instead of a viewport breakpoint. The sidebar and an adjacent editor can leave a narrow panel even on an iPad or laptop. Put `@container/workspace` on its parent and use `@min-[42rem]/workspace:flex-row` on the split region, then size the rail at the same container breakpoint. `ComplianceUI` is the existing example. Choose the threshold from the minimum usable widths of both panels.

An editing workspace can keep separate scroll regions on desktop. On a phone, the stacked panels must remain reachable through scrolling. Avoid a viewport-height calculation that subtracts a guessed header size. Let the shared shell and flex layout provide the available height. Use a modest minimum height for an editor after stacking so it does not collapse.

Comparisons can use `grid-cols-[repeat(auto-fit,minmax(min(100%,20rem),1fr))]` to retain useful panel widths. Do not shrink three panels into unreadable slivers. Make long code blocks scroll within their own region.

## Standard management page

This composition uses the actual shared API. Replace the sample labels with existing translated strings and wire the real handlers and table. Do not copy placeholder data into production.

```tsx
"use client";

import { useState, type ReactNode } from "react";
import { Page, PageContent, PageTabs, PageTabsContent, PageTabsList, PageTabsTrigger } from "@/components/shared/Page";
import { PageHeader, PageHeaderControls, PageHeaderDescription, PageHeaderTitle } from "@/components/shared/PageHeader";
import { Button } from "@/components/ui/button";

export function ResourcePage({
  resources,
  settings,
  onAdd,
  onImport,
}: {
  resources: ReactNode;
  settings: ReactNode;
  onAdd: () => void;
  onImport: () => void;
}) {
  const [tab, setTab] = useState("resources");
  return (
    <Page>
      <PageHeader>
        <PageHeaderTitle>Resources</PageHeaderTitle>
        <PageHeaderDescription>Manage the resources available to your team.</PageHeaderDescription>
        <PageHeaderControls>
          <Button onClick={onAdd}>Add resource</Button>
          <Button variant="outline" onClick={onImport}>
            Import
          </Button>
        </PageHeaderControls>
      </PageHeader>
      <PageTabs value={tab} onValueChange={(value) => setTab(String(value))}>
        <PageTabsList>
          <PageTabsTrigger value="resources">Resources</PageTabsTrigger>
          <PageTabsTrigger value="settings">Settings</PageTabsTrigger>
        </PageTabsList>
        <PageTabsContent value="resources">
          <PageContent>{resources}</PageContent>
        </PageTabsContent>
        <PageTabsContent value="settings">
          <div className="max-w-4xl space-y-6">{settings}</div>
        </PageTabsContent>
      </PageTabs>
    </Page>
  );
}
```

For a real page with deep links, retain its existing `useUrlTab` or `nuqs` setup instead of the local state used to keep this layout example self-contained.

## Verification contract

Check the actual route inside the full dashboard shell at all four sizes. A viewport includes the sidebar, not just the content column.

| View           | Dimensions  | Main check                                                      |
| -------------- | ----------- | --------------------------------------------------------------- |
| Widescreen     | 1920 × 1080 | Useful width, bounded reading lines, no oversized empty gutters |
| Laptop         | 1366 × 768  | Actions and workspace controls fit with the sidebar expanded    |
| iPad landscape | 1024 × 768  | Panels adapt before fields become cramped                       |
| Phone          | 390 × 844   | Readable content, reachable tabs and actions, no page overflow  |

Visit every affected tab, including nested tabs and settings accordions. Check the beginning and end of long pages. Open representative dialogs, dropdowns and date pickers. Use the keyboard to reach controls and scroll clipped tab labels into view. Verify both themes, a long name, populated data, an empty state, and a loading or error state where relevant.

Capture before and after at the same route, selected state, data, theme, sidebar state, and viewport. Record any difference in scroll position. Keep screenshots and fixtures outside the source change. Use synthetic or sanitized examples and never capture credentials or private infrastructure. Label fixture screenshots as synthetic visual QA, separate from live backend verification.

Run Prettier, focused lint and the relevant existing tests. Do not write tests that merely assert class strings. Compare a failing check with the base under equivalent conditions before calling it an introduced defect. Keep functional fixes separate from a styling-only change.

## Prompt for a UI agent

> Read this guide and the applicable AGENTS.md files. Inspect the existing route in the full shell before editing. Preserve its components, data, copy, actions and behavior. Choose the management, analytics, settings, workspace or detail layout that matches the task. Use shared page gutters, heading hierarchy, tabs and semantic tokens. Make narrow screens readable through wrapping, stacking and local scrolling. Verify every affected state at 1920×1080, 1366×768, 1024×768 and 390×844, including keyboard access, long content and both themes. Keep before/after screenshots outside the repository. Report exactly what was exercised and any remaining gap.
