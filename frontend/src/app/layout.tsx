import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Navbar from "@/components/Navbar";
import "react-toastify/dist/ReactToastify.css";
import "@mantine/core/styles.css";

import { ColorSchemeScript, MantineProvider } from "@mantine/core";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Garbage Classifier",
  icons: {icon: "/logo.svg"}
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <ColorSchemeScript />
      </head>
      <body className={inter.className}>
        <div className="px-[5%] flex flex-col min-h-screen">
          <Navbar />
          <MantineProvider>{children}</MantineProvider>
        </div>
      </body>
    </html>
  );
}
