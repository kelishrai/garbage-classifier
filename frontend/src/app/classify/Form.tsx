"use client";

import { Api } from "@/api/api";
import Button from "@/components/ui/Button";
import Image from "next/image";
import React, { useRef, useState } from "react";
import { toast } from "react-toastify";

const delay = (ms: number) => new Promise((res) => setTimeout(res, ms));

const Form = ({ setResult }: { setResult: any }) => {
  const fileRef = React.useRef<HTMLInputElement>(null);
  const [fileName, setFileName] = useState("");
  const [loading, setLoading] = useState(false);
  const toastId = useRef();

  const dismiss = () => toast.dismiss(toastId.current);

  const submitForm = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const file = fileRef.current?.files?.[0];
    if (!file) {
      return;
    }
    const formData = new FormData();
    formData.append("garbage", file);
    const api = new Api();
    setLoading(true);
    const res = await api
      .request({
        method: "POST",
        url: "http://127.0.0.1:8000/get-prediction",
        body: formData,
      })
      .catch(async (err) => {
        const msg = err?.response?.data?.garbage?.[0];
        const newMsg = msg?.split(".")?.[0];

        toast.error(newMsg || "An error occurred");
        setResult(null);
        await delay(500);
        setLoading(false);
      });
    await delay(500);
    if (res) {
      setLoading(false);
      dismiss();
      toast.success("Success");
      setResult(res?.data);
    }
  };

  return (
    <form
      onSubmit={submitForm}
      className="flex gap-3 w-full justify-center h-34"
    >
      <label htmlFor="underline_select" className="sr-only">
        Underline select
      </label>
      <div className="relative">
        <svg
          width="32"
          height="32"
          viewBox="0 0 32 32"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
          className="absolute top-1/2 -translate-y-1/2 right-2"
        >
          <path
            d="M23.8933 10.9066H15.5867H8.10667C6.82667 10.9066 6.18667 12.4533 7.09333 13.3599L14 20.2666C15.1067 21.3733 16.9067 21.3733 18.0133 20.2666L20.64 17.6399L24.92 13.3599C25.8133 12.4533 25.1733 10.9066 23.8933 10.9066Z"
            fill="white"
          />
        </svg>
      </div>

      <div className="flex items-center justify-center w-fit h-full">
        <label
          htmlFor="dropzone-file"
          className="flex flex-col items-center justify-center py-4 pb-2 px-5 bg-primary rounded-2xl cursor-pointer border border-primary shadow-md"
        >
          <div className="flex items-center justify-center gap-2">
            <p className="text-[20px] text-primary-text font-normal min-w-32 pb-2">
              {fileName || "Upload Image"}
            </p>
            <Image
              src="/file.svg"
              width={40}
              height={40}
              alt="upload"
              className="text-white"
            />
          </div>
          <input
            id="dropzone-file"
            type="file"
            className="hidden"
            onChange={(e) => setFileName(e?.target?.files?.[0]?.name as any)}
            ref={fileRef}
          />
        </label>
      </div>
      <button
        className="text-[20px] bg-gradient-to-r min-w-32 bg-base-primary text-white hover:text-white rounded-2xl transition-all shadow-md flex justify-center items-center py-4 px-5"
        type="submit"
      >
        {loading ? (
          <div className="h-5 w-5 animate-spin border-[3px] border-white border-r-primary rounded-full"></div>
        ) : (
          <span>Classify</span>
        )}
      </button>
    </form>
  );
};

export default Form;
