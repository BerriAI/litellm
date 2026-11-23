import { Fragment, type MouseEvent } from "react";
import { useTranslation } from "react-i18next";
import {
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
import { useCustomerHierarchyLinks } from "./CustomerHierarchyNavigation";
import type { CustomerUsageScope } from "./customerUsageApi";

function navigate(event: MouseEvent<HTMLAnchorElement>, onClick?: () => void) {
  if (!onClick || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
  event.preventDefault();
  onClick();
}

export default function CustomerHierarchyBreadcrumbs({
  scope,
  page,
  organizationLabel,
  teamLabel,
  memberLabel,
  onOrganizationClick,
  onTeamClick,
}: {
  scope: CustomerUsageScope;
  page: "usage" | "manage";
  organizationLabel?: string | null;
  teamLabel?: string | null;
  memberLabel?: string | null;
  onOrganizationClick: () => void;
  onTeamClick: () => void;
}) {
  const { t } = useTranslation();
  const { customersHref, customerHierarchyHref } = useCustomerHierarchyLinks();
  const items = [
    { label: t("customerPortal.analytics.allCustomers"), href: customersHref, onClick: undefined },
    {
      label: organizationLabel || scope.org,
      href: customerHierarchyHref(page, { org: scope.org, team: null, member: null }),
      onClick: onOrganizationClick,
    },
    ...(scope.team
      ? [
          {
            label: teamLabel || scope.team,
            href: customerHierarchyHref(page, { org: scope.org, team: scope.team, member: null }),
            onClick: onTeamClick,
          },
        ]
      : []),
    ...(scope.member
      ? [{ label: memberLabel || scope.member, href: customerHierarchyHref(page, scope), onClick: undefined }]
      : []),
  ];
  return (
    <BreadcrumbList className="gap-2 text-xs">
      {items.map((item, index) => (
        <Fragment key={item.href}>
          {index > 0 && <BreadcrumbSeparator />}
          <BreadcrumbItem>
            {index === items.length - 1 ? (
              <BreadcrumbPage>{item.label}</BreadcrumbPage>
            ) : (
              <BreadcrumbLink href={item.href} onClick={(event) => navigate(event, item.onClick)}>
                {item.label}
              </BreadcrumbLink>
            )}
          </BreadcrumbItem>
        </Fragment>
      ))}
    </BreadcrumbList>
  );
}
