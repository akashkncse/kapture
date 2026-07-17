"use client";

import { FormEvent, useState } from "react";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [response, setResponse] = useState<string>("");
  const [error, setError] = useState<string>("");
  const [loading, setLoading] = useState(false);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!file) {
      setError("Select a PDF file first.");
      setResponse("");
      return;
    }

    setLoading(true);
    setError("");
    setResponse("");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const apiResponse = await fetch("http://localhost:8000/extract-syllabus/", {
        method: "POST",
        body: formData,
      });

      const data = await apiResponse.json();

      if (!apiResponse.ok) {
        setError(JSON.stringify(data, null, 2));
        return;
      }

      setResponse(JSON.stringify(data, null, 2));
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <form onSubmit={handleSubmit}>
        <input
          type="file"
          accept="application/pdf"
          onChange={(event) => setFile(event.target.files?.[0] ?? null)}
        />
        <button type="submit" disabled={loading}>
          {loading ? "Uploading..." : "Upload"}
        </button>
      </form>

      {error ? <pre>{error}</pre> : null}
      {response ? <pre>{response}</pre> : null}
    </main>
  );
}
