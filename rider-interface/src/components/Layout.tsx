
import React from "react";
import { Outlet } from "react-router-dom";
import Header from "./Header";
import { cn } from "@/lib/utils";

interface LayoutProps {
  className?: string;
  children?: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ className, children }) => {
  return (
    <div className="flex flex-col min-h-screen bg-background">
      <Header />
      <main 
        className={cn(
          "flex-1 container mx-auto px-4 pt-4 pb-20 animate-fade-in",
          className
        )}
      >
        {children || <Outlet />}
      </main>
    </div>
  );
};

export default Layout;
