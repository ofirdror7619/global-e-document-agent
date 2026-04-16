import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Document Agent UI",
  description: "Upload files and ask questions",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
