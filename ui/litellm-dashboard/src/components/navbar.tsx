import Link from "next/link";
import React, { useState, useEffect } from "react";
import type { MenuProps } from "antd";
import { Dropdown } from "antd";
import { getProxyBaseUrl, Organization } from "@/components/networking";
import { defaultOrg } from "@/components/common_components/default_org";
import { 
  UserOutlined,
  LogoutOutlined
} from '@ant-design/icons';
import { clearTokenCookies } from "@/utils/cookieUtils";
import { fetchProxySettings } from "@/utils/proxyUtils";

interface NavbarProps {
  userID: string | null;
  userEmail: string | null;
  userRole: string | null;
  premiumUser: boolean;
  setProxySettings: React.Dispatch<React.SetStateAction<any>>;
  proxySettings: any;
  accessToken: string | null;
}

const Navbar: React.FC<NavbarProps> = ({
  userID,
  userEmail,
  userRole,
  premiumUser,
  proxySettings,
  setProxySettings,
  accessToken,
}) => {
  const baseUrl = getProxyBaseUrl();
  const imageUrl = baseUrl + "/get_image";
  const [logoutUrl, setLogoutUrl] = useState("");

  useEffect(() => {
    const initializeProxySettings = async () => {
      if (accessToken) {
        const settings = await fetchProxySettings(accessToken);
        console.log("response from fetchProxySettings", settings);
        if (settings) {
          setProxySettings(settings);
        }
      }
    };

    initializeProxySettings();
  }, [accessToken]);

  useEffect(() => {
    setLogoutUrl(proxySettings?.PROXY_LOGOUT_URL || "");
  }, [proxySettings]);

  const handleLogout = () => {
    clearTokenCookies();
    window.location.href = logoutUrl;
  };

  const userItems: MenuProps["items"] = [
    {
      key: "1",
      label: (
        <div className="py-1">
          <p className="text-sm text-gray-600">Role: {userRole}</p>
          <p className="text-sm text-gray-600">Email: {userEmail || "Unknown"}</p>
          <p className="text-sm text-gray-600"><UserOutlined /> {userID}</p>
          <p className="text-sm text-gray-600">Premium User: {String(premiumUser)}</p>
        </div>
      ),
    },
    {
      key: "2",
      label: <p className="text-sm hover:text-gray-900" onClick={handleLogout}><LogoutOutlined /> Logout</p>,
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
