"use client";

import { useEffect, useState } from "react";
import { getPublicModelHubInfo } from "@/components/networking";

/**
 * Client component that fetches the public model hub info and sets
 * a custom favicon if one is configured.
 */
export default function FaviconSetter() {
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    const setFavicon = async () => {
      try {
        const info = await getPublicModelHubInfo();
        if (info.favicon_url) {
          // Find existing favicon link element or create a new one
          let link: HTMLLinkElement | null = document.querySelector("link[rel*='icon']");
          
          if (!link) {
            link = document.createElement("link");
            link.rel = "icon";
            document.head.appendChild(link);
          }
          
          link.href = info.favicon_url;
        }
      } catch (error) {
        // Silently fail - fallback to default favicon
        console.debug("Could not fetch public model hub info for favicon:", error);
      } finally {
        setIsLoaded(true);
      }
    };

    setFavicon();
  }, []);

  // This component doesn't render anything
  return null;
}