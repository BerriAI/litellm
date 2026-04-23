# Phase 1 migration blueprint

This document is finalized after Section 1 (Access Groups) and locked for the
rest of the phase-1 run. Later sections follow the patterns defined here; any
deviation is logged in `DEVIATIONS.md`.

**Status:** Not yet locked — will be drafted during Section 1 (Access Groups).

Once locked, this file will contain:

1. **Component translation table** — every antd component → shadcn primitive,
   with import paths and any prop-shape differences.
2. **Icon translation table** — @ant-design/icons, @heroicons/react,
   @remixicon/react → lucide-react (only).
3. **Toast pattern** — before/after code example using the
   `MessageManager` / `NotificationManager` bridges (which now delegate to
   sonner under the hood).
4. **Form pattern** — canonical shadcn `<Form>` + `react-hook-form` + `zod`
   example, including schema location, `FormField` / `FormItem` / `FormLabel`
   / `FormControl` / `FormDescription` / `FormMessage` wiring, and submit
   handler hooked into the existing data-layer function.
5. **Table pattern** — shadcn `<Table>` + `@tanstack/react-table` with
   sort / filter / pagination. Column-visibility UI if used.
6. **Shared layout patterns** — page header, section card, empty state,
   loading skeleton, error alert, permissions gate.
