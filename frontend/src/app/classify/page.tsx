"use client";

import React, { useState } from "react";
import Form from "./Form";
import { ToastContainer } from "react-toastify";
import Result from "./result";
import Animation from "./Animation";

const Page = () => {
  const [result, setResult] = useState(null);

  return (
    <section className="ml-3 w-full gap-0 flex flex-col pr-5">
      <p className="text-3xl font-extralight">
        Ready for some{" "}
        <span className="text-primary font-bold">amazing classification?</span>
      </p>
      <div className="h-full w-full mt-10">
        {result ? <Result result={result} /> : <Animation />}
      </div>
      <div className="h-6 w-full" />
      <div className="h-2 w-full"></div>
      <Form {...{ setResult }} />
      <div className="h-2 w-full"></div>
      <p className="text-sm text-base-primary font-extralight mx-auto">
        Upload an image then click the wand icon
      </p>
      <ToastContainer autoClose={2000} hideProgressBar pauseOnHover />
    </section>
  );
};

export default Page;
