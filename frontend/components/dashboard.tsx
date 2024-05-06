'use client'
/**
* This code was generated by v0 by Vercel.
* @see https://v0.dev/t/FDVtOzeVtqa
* Documentation: https://v0.dev/docs#integrating-generated-code-into-your-nextjs-app
*/
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import Uppy from '@uppy/core';
import { Dashboard } from '@uppy/react';
import XHR from '@uppy/xhr-upload';
import { useState } from "react";

import '@uppy/core/dist/style.min.css';
import '@uppy/dashboard/dist/style.min.css';

export function DashboardComponent() {
  const [question, setQuestion] = useState('');
  const [uppy] = useState(() => new Uppy({
    restrictions: { allowedFileTypes: ['application/pdf'] },

  })
    .use(XHR, {
      endpoint: 'http://127.0.0.1:8000/upload/',
      bundle: true,
      formData: true,
      fieldName: 'files',
    })
  )

  return (
    <div className="flex h-screen bg-[#18181b] text-white">
      <div className="flex flex-col w-[300px] border-r border-gray-800 p-4">
        <Dashboard
          uppy={uppy}
          proudlyDisplayPoweredByUppy={false}
          hideUploadButton
        />
        <Button
          className="mt-4"
          disabled={question.length == 0 || uppy.getState().uploading}
          onClick={() => {
            console.log('Uploading Question...');
            fetch('http://127.0.0.1:8000/question/', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({ ques: question }),
            })
              .then(res => res.json())
              .then(data => {
                console.log(data);
                uppy.upload();
              })
          }}
        >
          Process
        </Button>
      </div>
      <div className="flex-1 p-4">
        <div className="flex justify-between pb-4">
          <h1 className="text-3xl font-bold">RagOllama - Adaptive RAG Chatbot</h1>
        </div>
        <div className="flex flex-col items-start py-4">
          <label className="mb-2" htmlFor="chat-question">
            Question:
          </label>
          <Input
            className="bg-gray-700" id="chat-question"
            placeholder="Ask about your PDF"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            disabled={uppy.getState().uploading}
          />
        </div>
      </div>
    </div>
  )
}
