import Button from "@/components/ui/Button";
import Image from "next/image";

export default function Home() {
  return (
    <main className="flex flex-1 flex-col relative">
      <div className="flex flex-col gap-6 pl-2">
        <p className="text-3xl text-base-primary font-extralight">
          Garbage classification made easy
        </p>
        <h1 className="text-5xl leading-normal font-extralight text uppercase">
          Your{" "}
          <span className="text-primary font-bold">
            Garbage <br /> Classifier
          </span>{" "}
          Buddy
        </h1>
        <p className="text-base-primary font-extralight">
          The only AI model you need to accurately analyze and categorize
          <br />
          waste items in real-time.
        </p>
        <p className="text-base-primary font-extralight">
          Created using advanced cutting-edge machine learning
          <br />
          algorithms.
        </p>
      </div>
      <Image
        src="/landing.svg"
        width={600}
        height={500}
        alt="landing"
        className="absolute bottom-0 right-0"
      />
    </main>
  );
}
