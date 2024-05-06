import { ArrowUpRight } from "lucide-react";
import Link from "next/link";

export function Footer() {
    return (
        <footer className="w-full bg-gray-100 py-6 dark:bg-gray-800 relative bottom-0 left-0 right-0 z-10">
            <div className="container px-4 md:px-6 flex flex-col md:flex-row items-center justify-between">
                <p className="text-sm text-gray-500 dark:text-gray-400">© {new Date().getFullYear()} RagOllama. All rights reserved.</p>
                <div className="flex gap-2 mt-4 md:mt-0">
                    Made with ❤️ by
                    <Link
                        href='https:github.com/CantBeSubh'
                        className="text-blue-400 inline-flex">
                        CantBeSubh
                        <ArrowUpRight className="text-base" />
                    </Link>
                </div>
            </div>
        </footer>
    )
}