import { TabsContent, TabsTrigger } from "@/components/ui/tabs";
import type { DateRangePickerValue } from "@/components/shared/date_picker_types";
import { isProxyAdminRole } from "@/utils/roles";
import CategoryROI from "./CategoryROI";

type Access = { role: string | null; view: string };
const permitted = ({ role, view }: Access) => isProxyAdminRole(role ?? "") && view === "global";

export function ROIUsageTrigger(props: Access) {
  if (!permitted(props)) return null;
  return (
    <TabsTrigger value="roi" className="flex-none px-3">
      ROI
    </TabsTrigger>
  );
}

export function ROIUsageContent(
  props: Access & { tab: string; token: string | null; range: DateRangePickerValue; user: string | null },
) {
  if (!permitted(props) || props.tab !== "roi" || !props.token) return null;
  if (!props.range.from || !props.range.to) return null;
  return (
    <TabsContent value="roi">
      <CategoryROI
        token={props.token}
        start={props.range.from.toISOString()}
        end={props.range.to.toISOString()}
        user={props.user ?? ""}
      />
    </TabsContent>
  );
}
