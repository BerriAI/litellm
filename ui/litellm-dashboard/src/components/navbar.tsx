import Link from "next/link";
import React from "react";
import type { MenuProps } from "antd";
import { Dropdown } from "antd";
import { CogIcon } from "@heroicons/react/outline";
import { Organization } from "@/components/networking";

interface NavbarProps {
  userID: string | null;
  userRole: string | null;
  userEmail: string | null;
  premiumUser: boolean;
  setProxySettings: React.Dispatch<React.SetStateAction<any>>;
  proxySettings: any;
  currentOrg: Organization | null;
  onOrgChange?: (org: Organization) => void;
  onNewOrg?: () => void;
  organizations?: Organization[];
}

const Navbar: React.FC<NavbarProps> = ({
  userID,
  userRole,
  userEmail,
  premiumUser,
  setProxySettings,
  proxySettings,
  currentOrg = null,
  onOrgChange = () => {},
  onNewOrg = () => {},
  organizations = []
}) => {
  console.log(`currentOrg: ${JSON.stringify(currentOrg)}`)
  console.log(`organizations: ${JSON.stringify(organizations)}`)
  const isLocal = process.env.NODE_ENV === "development";
  const imageUrl = isLocal ? "http://localhost:4000/get_image" : "/get_image";
  let logoutUrl = proxySettings?.PROXY_LOGOUT_URL || "";

  const handleLogout = () => {
    document.cookie = "token=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;";
    window.location.href = logoutUrl;
  };

  const userItems: MenuProps["items"] = [
    {
      key: "1",
      label: (
        <div className="px-1 py-1">
          <p className="text-sm text-gray-600">Role: {userRole}</p>
          <p className="text-sm text-gray-600">ID: {userID}</p>
          <p className="text-sm text-gray-600">Premium User: {String(premiumUser)}</p>
        </div>
      ),
    },
    {
      key: "2",
      label: <p className="text-sm hover:text-gray-900" onClick={handleLogout}>Logout</p>,
    }
  ];

  const orgMenuItems: MenuProps["items"] = [
    {
      key: 'header',
      label: 'Organizations',
      type: 'group' as const,
      style: { 
        color: '#6B7280',
        fontSize: '0.875rem'
      }
    },
    {
      key: "default",
      label: (
        <div className="flex items-center justify-between py-1">
          <span className="text-sm">Default Organization</span>
        </div>
      ),
      onClick: () => onOrgChange({ 
        organization_id: null, 
        organization_alias: "Default Organization" 
      } as Organization)
    },
    ...organizations.filter(org => org.organization_id !== null).map(org => ({
      key: org.organization_id ?? "default",
      label: (
        <div className="flex items-center justify-between py-1">
          <span className="text-sm">{org.organization_alias}</span>
        </div>
      ),
      onClick: () => onOrgChange(org)
    })),
    {
      key: "note",
      label: (
        <div className="flex items-center justify-between py-1 px-2 bg-gray-50 text-gray-500 text-xs italic">
          <span>Switching between organizations on the UI is currently in beta.</span>
        </div>
      ),
      disabled: true
    }
  ];

  return (
    <nav className="bg-white border-b border-gray-200 sticky top-0 z-10">
      <div className="w-full">
        <div className="flex items-center h-12 px-4">
          {/* Left side with correct logo positioning */}
          <div className="flex items-center flex-shrink-0">
            <Link href="/" className="flex items-center">
              <img
                src={imageUrl}
                alt="LiteLLM Brand"
                className="h-8 w-auto"
              />
            </Link>
          </div>

          {/* Organization selector with beta label */}
          <div className="ml-8 flex items-center">
            <Dropdown 
              menu={{ 
                items: orgMenuItems,
                style: {
                  width: '280px',
                  padding: '4px',
                  marginTop: '4px'
                }
              }}
              trigger={['click']}
              placement="bottomLeft"
            >
              <button className="inline-flex items-center text-[13px] text-gray-700 hover:text-gray-900 transition-colors">
                {currentOrg ? currentOrg.organization_alias : "Default Organization"}
                <svg
                  className="ml-1 w-4 h-4 text-gray-500"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 9l-7 7-7-7" />
                </svg>
              </button>
            </Dropdown>
            <span className="ml-2 text-[10px] bg-blue-100 text-blue-800 font-medium px-2 py-0.5 rounded">BETA</span>
          </div>

          {/* Right side nav items */}
          <div className="flex items-center space-x-5 ml-auto">
            <a
              href="https://docs.litellm.ai/docs/"
              target="_blank"
              rel="noopener noreferrer"
              className="text-[13px] text-gray-600 hover:text-gray-900 transition-colors"
            >
              Docs
            </a>

            <Dropdown 
              menu={{ 
                items: userItems,
                style: {
                  padding: '4px',
                  marginTop: '4px'
                }
              }}
            >
              <button className="inline-flex items-center text-[13px] text-gray-600 hover:text-gray-900 transition-colors">
                User
                <svg
                  className="ml-1 w-4 h-4 text-gray-500"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 9l-7 7-7-7" />
                </svg>
              </button>
            </Dropdown>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;