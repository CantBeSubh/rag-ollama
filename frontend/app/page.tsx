import { Footer } from "@/components/footer";
import { LandingPage } from "@/components/landing-page";

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-between">
      <LandingPage />
      <Footer />
    </main>
  );
}
