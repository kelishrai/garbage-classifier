import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import Image from "next/image";

const Training = () => {
    const tabs = ["ResNet18", "ResNet34", "ResNet50"];

    const imgPaths = [
        "accuracy.png",
        "confusion_matrix.png",
        "loss_batch.png",
        "loss.png",
        "lr.png",
    ];

    return (
        <main className="flex flex-1 flex-col pl-3 relative justify-start">
            <div className="flex flex-col">
                <h1 className="text-3xl font-extralight mb-8">
                    Training Results of the{" "}
                    <span className="text-primary font-bold"> ResNet</span>{" "}
                    Models
                </h1>

                <Tabs defaultValue="resnet18" className="w-full">
                    <TabsList>
                        {tabs.map((tab, index) => (
                            <TabsTrigger key={tab} value={tab.toLowerCase()}>
                                {tab}
                            </TabsTrigger>
                        ))}
                    </TabsList>
                    {tabs.map((tab, index) => (
                        <TrainingTabContent
                            key={tab}
                            tabKey={tab.toLowerCase()}
                            images={imgPaths}
                        />
                    ))}
                </Tabs>
            </div>
        </main>
    );
};

export default Training;

const TrainingTabContent = ({
    tabKey,
    images,
}: {
    tabKey: string;
    images: string[];
}) => {
    return (
        <TabsContent value={tabKey} className="grid gap-4 grid-cols-2">
            <Image
                src="/dataset.png"
                width={1000}
                height={800}
                alt="accuracy"
                className="mt-36"
            />
            {images.map((imgPath) => (
                <Image
                    key={imgPath}
                    src={`${process.env.NEXT_PUBLIC_APP_DOMAIN}/static/${tabKey}/${imgPath}`}
                    width={1000}
                    height={500}
                    alt="accuracy"
                />
            ))}
        </TabsContent>
    );
};
