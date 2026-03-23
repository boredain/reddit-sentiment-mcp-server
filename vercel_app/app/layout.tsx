import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Customer Insights",
  description: "Reddit-powered customer insights via Vercel AI SDK",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-black text-white min-h-screen antialiased">{children}</body>
    </html>
  );
}
