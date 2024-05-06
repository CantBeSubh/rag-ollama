import { BackgroundGradientAnimation } from "@/components/ui/background-gradient-animation";
import { SignIn } from "@clerk/nextjs";
import { dark } from "@clerk/themes";

export default function Page() {
    return (
        <BackgroundGradientAnimation className="">
            <div className=" absolute top-[50%] left-[50%] -translate-x-1/2 -translate-y-1/2 z-10">
                <SignIn path="/sign-in" appearance={{ baseTheme: dark }} />
            </div>
        </BackgroundGradientAnimation>
    )
}