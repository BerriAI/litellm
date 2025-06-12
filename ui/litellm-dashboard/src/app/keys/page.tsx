"use client";

import { cx } from "@/lib/cva.config";
import { VirtualKeyTable } from "./virtual-keys-table";
import { KeyRoundIcon } from "lucide-react";

export default function KeysPage() {
  return (
    <div className="p-8 h-full flex flex-col min-h-0">
      <div className="flex mb-8 shrink-0 flex-col gap-2">
        <div className="flex items-center gap-3">
          <div
            className={cx(
              "size-[48px] shrink-0 flex justify-center items-center bg-indigo-50 rounded-lg",
              "border border-indigo-100",
            )}
          >
            <KeyRoundIcon className="size-6 text-indigo-600" />
          </div>

          <h1 className="_font-medium text-[32px] tracking-tighter text-neutral-900">
            Virtual Keys Management
          </h1>
        </div>

        <p className="max-w-[480px] text-[15px]/[1.6] text-neutral-500 tracking-tight">
          Manage and monitor your API keys with secure access control and
          real-time usage insights
        </p>
      </div>

      <VirtualKeyTable />
    </div>
  );
}
