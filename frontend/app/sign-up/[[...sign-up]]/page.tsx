import { BackgroundGradientAnimation } from "@/components/ui/background-gradient-animation";
import { SignUp } from "@clerk/nextjs";

export default function Page() {
    return (
        <BackgroundGradientAnimation>
            <div className=" absolute top-[50%] left-[50%] -translate-x-1/2 -translate-y-1/2 z-10">
                <SignUp path="/sign-up" />
            </div>
        </BackgroundGradientAnimation>
    )
}