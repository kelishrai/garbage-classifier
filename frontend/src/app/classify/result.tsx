import React, { useState } from "react";
import Image from "next/image";

const models = [
  {
    name: "ResNet18",
    key: "prediction_result_18",
  },
  {
    name: "ResNet34",
    key: "prediction_result_34",
  },
  {
    name: "ResNet50",
    key: "prediction_result_50",
  },
];

const result = ({ result }: { result: any }) => {
  const recyclableItems = [
    "battery",
    "brown-glass",
    "cardboard",
    "clothes",
    "green-glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "white-glass",
  ];

  return (
    <div className="bg-gray w-full h-[340px] rounded-2xl flex flex-row items-center justify-start gap-10 p-10">
      <div className="w-[400px] h-full rounded-2xl overflow-hidden flex flex-row items-center justify-center">
        <Image
          src={result["image"]}
          alt="result"
          width={300}
          height={300}
          className="w-full h-full"
        />
      </div>
      <div className="h-full flex-col justify-around w-full">
        <div className="flex flex-row items-center justify-between ml-20 w-full">
          <div className="flex-1 text-left">
            <p className="text-base-primary text-3xl font-light">Model</p>
          </div>
          <div className="flex-1 text-left">
            <p className="text-base-primary text-3xl font-light">
              Class
            </p>
          </div>
          <div className="flex-1 text-left">
            <p className="text-base-primary text-3xl font-light">
              Category
            </p>
          </div>
        </div>
        <div className="w-full bg-white h-1 rounded-sm my-10"></div>
        <div className="flex flex-col gap-4 w-full">
          {models.map((model) => (
            <div
              key={model.name}
              className="flex flex-row items-center justify-between ml-20 w-full"
            >
              <div className="flex-1 text-left">
                <p className="text-primary text-2xl font-bold">{model.name}</p>
              </div>
              <div className="flex-1 text-left">
                <p className="text-primary text-2xl font-bold capitalize">
                  {result[model.key]}
                </p>
              </div>
              <div className="flex-1 text-left">
                <p className="text-primary text-2xl font-bold">
                  {recyclableItems.includes(result[model.key])
                    ? "Recyclable"
                    : "Non-Recyclable"}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default result;
