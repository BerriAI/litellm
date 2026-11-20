import { Barlow_Condensed, DM_Sans, IBM_Plex_Mono } from "next/font/google";

import styles from "./catalog.module.css";

const sans = DM_Sans({ subsets: ["latin"], variable: "--font-catalog-sans" });
const mono = IBM_Plex_Mono({ subsets: ["latin"], weight: "400", variable: "--font-catalog-mono" });
const display = Barlow_Condensed({ subsets: ["latin"], weight: "600", variable: "--font-catalog-display" });

export const catalogTypography = `${styles.typography} ${sans.variable} ${mono.variable} ${display.variable}`;
