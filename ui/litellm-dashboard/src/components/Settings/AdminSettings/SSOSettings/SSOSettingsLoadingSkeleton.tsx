"use client";

import { Card } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Shield } from "lucide-react";

export default function SSOSettingsLoadingSkeleton() {
  return (
    <Card className="p-6">
      <div className="flex flex-col gap-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Shield className="w-6 h-6 text-muted-foreground" />
            <div>
              <h3 className="text-lg font-semibold m-0">SSO Configuration</h3>
              <p className="text-muted-foreground">
                Manage Single Sign-On authentication settings
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <Skeleton className="h-8 w-[170px]" />
            <Skeleton className="h-8 w-[190px]" />
          </div>
        </div>

        <div className="border border-border rounded-md overflow-hidden">
          {[100, 200, 250, 180, 220].map((w, i) => (
            <div
              key={i}
              className={`grid grid-cols-[minmax(120px,200px)_1fr] ${
                i !== 0 ? "border-t border-border" : ""
              }`}
            >
              <div className="bg-muted px-4 py-2.5">
                <Skeleton className="h-4 w-20" />
              </div>
              <div className="px-4 py-2.5">
                <Skeleton className="h-4" style={{ width: w }} />
              </div>
            </div>
          ))}
        </div>
      </div>
    </Card>
  );
}
