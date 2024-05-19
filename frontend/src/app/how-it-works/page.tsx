import React from "react";
import Image from "next/image";
import { useState } from "react";
import { Stepper, StepperProps, rem } from "@mantine/core";

const page = () => {
  return (
    <main className="flex flex-1 flex-col pl-3 relative justify-start">
      <div className="flex flex-col">
        <h1 className="text-3xl font-extralight">
          It&apos;s really{" "}
          <span className="text-primary font-bold"> easy to use</span> YGBC
        </h1>

        <div className="flex flex-row w-full mt-10 gap-[200px] h-[320px]">
          <div className="absolute top-[14%] w-[76%] left-[12%] h-1 rounded-md bg-gray z-[-1]"></div>
          <LineItem
            index={1}
            imageSrc="/Step 1.jpg"
            description="Click the “Let’s Classify” button"
          />
          <LineItem
            index={2}
            imageSrc="/Step 2.jpg"
            description="Upload the image you want to classify
            and click the wand icon"
          />
          <LineItem
            index={3}
            imageSrc="/Step 3.jpg"
            description="View your result"
          />
        </div>
        <div>
          <h1 className="text-3xl leading-normal font-extralight mt-8">
            <span className="text-primary font-bold">Thanks</span> for checking
            us out!
          </h1>
          <p className="text-md font-extralight mt-5">
            We aim to automate the process of waste classification
            <br />
            thereby enforcing effective procedures while also reducing the
            <br />
            cost and labor that goes into manual garbage sorting
          </p>
        </div>
      </div>
    </main>
  );
};

export default page;

const LineItem = ({
  index,
  imageSrc,
  description,
}: {
  index: number;
  imageSrc: string;
  description: string;
}) => {
  return (
    <div className="flex-1">
      <div className="flex flex-col justify-between items-center h-[300px]">
        <div className="text-white w-6 h-6 bg-gray rounded-full text-center">
          {index}
        </div>
        <div className="text-md font-extralight text-center">{description}</div>
        <Image
          alt="Steps Image"
          src={imageSrc}
          width={400}
          height={200}
          className="rounded-2xl shadow-md w-full h-auto"
        />
      </div>
    </div>
  );
};
