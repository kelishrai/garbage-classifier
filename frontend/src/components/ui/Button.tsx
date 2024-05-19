import clsx from "clsx";
import React from "react";

type ButtonType = {
  children: React.ReactNode;
  appearance: "primary" | "secondary";
  onClick?: React.MouseEventHandler<HTMLButtonElement>;
  type?: "button" | "reset";
};

const buttonType = {
  primary: {
    bg: "bg-primary hover:bg-white hover:text-primary border border-primary",
    text: "text-white font-normal text-md hover:text-primary",
  },
  secondary: {
    bg: "bg-white hover:bg-primary hover:text-white border border-base-primary",
    text: "text-base-primary text-md hover:text-white",
  },
};

const Button = ({
  children,
  appearance,
  onClick,
  type = "button",
}: ButtonType) => {
  return (
    <button
      className={clsx(
        "rounded-2xl transition-all shadow-md flex justify-center items-center  w-40 h-12",
        buttonType[appearance].bg,
        buttonType[appearance].text
      )}
      onClick={onClick}
      type={type}
    >
      {children}
    </button>
  );
};

export default Button;
