"use client";

import Image from "next/image";
import React from "react";
import Button from "./ui/Button";
import { useRouter } from "next/navigation";
import Link from "next/link";

const Navbar = () => {
  const router = useRouter();

  return (
    <div className="w-full flex justify-between pt-12 pb-10 pr-2">
      <Link href="/">
        <Image
          alt="logo"
          src="/logo.svg"
          width={100}
          height={100}
          className="w-16 h-auto"
        />
      </Link>
      <div className="flex gap-3 h-16">
        <Button
          appearance="secondary"
          onClick={() => router.push("/training-result")}
        >
          Training Results
        </Button>
        <Button appearance="primary" onClick={() => router.push("/classify")}>
          Lets Classify
        </Button>
      </div>
    </div>
  );
};

export default Navbar;
