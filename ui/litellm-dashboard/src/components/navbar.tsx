"use client";

import Link from "next/link";
import Image from "next/image";
import React, { useState } from "react";
import type { MenuProps } from 'antd';
import { Dropdown, Space } from 'antd';
import { useSearchParams } from "next/navigation";
import {
  Button,
  Text,
  Metric,
  Title,
  TextInput,
  Grid,
  Col,
  Card,
} from "@tremor/react";

// Define the props type
interface NavbarProps {
  userID: string | null;
  userRole: string | null;
  userEmail: string | null;
  showSSOBanner: boolean;
}
const Navbar: React.FC<NavbarProps> = ({
  userID,
  userRole,
  userEmail,
  showSSOBanner,
}) => {
  console.log("User ID:", userID);
  console.log("userEmail:", userEmail);
  console.log("showSSOBanner:", showSSOBanner);

  // const userColors = require('./ui_colors.json') || {};
  const isLocal = process.env.NODE_ENV === "development";
  const imageUrl = isLocal ? "http://localhost:4000/get_image" : "/get_image";

  const items: MenuProps['items'] = [
    {
      key: '1',
      label: (
        <>
          <p>Role: {userRole}</p>
          <p>ID: {userID}</p>
        </>
      ),
    },
  ];

  return (
    <nav className="left-0 right-0 top-0 flex justify-between items-center h-12 mb-4">
      <div className="text-left my-2 absolute top-0 left-0">
        <div className="flex flex-col items-center">
          <Link href="/">
            <button className="text-gray-800 rounded text-center">
              <img
                src={imageUrl}
                width={160}
                height={160}
                alt="LiteLLM Brand"
                className="mr-2"
              />
            </button>
          </Link>
        </div>
      </div>
      <div className="text-right mx-4 my-2 absolute top-0 right-0 flex items-center justify-end space-x-2">
      {showSSOBanner ? (
          
        <div style={{
          // border: '1px solid #391085',
          padding: '6px',
          borderRadius: '8px', // Added border-radius property
        }}
      >
          <a
            href="https://calendly.com/d/4mp-gd3-k5k/litellm-1-1-onboarding-chat"
            target="_blank"
            style={{
              "fontSize": "14px",
              "textDecoration": "underline"
            }}
          >
            Request hosted proxy
          </a>
          </div>
        ) : null}

        <div style={{
            border: '1px solid #391085',
            padding: '6px',
            borderRadius: '8px', // Added border-radius property
          }}
        >
       <Dropdown menu={{ items }} >
            <Space>
              {userEmail}
            </Space>
        </Dropdown>
        </div>
        </div>

    </nav>
  );
};

export default Navbar;
