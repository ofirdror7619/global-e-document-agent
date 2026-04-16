import { UploadAndAsk } from "../components/upload-and-ask";

export default function HomePage() {
  return (
    <main className="page-shell">
      <header className="hero">
        <p className="eyebrow">Global-E Home Assignment</p>
        <h1>Document Agent</h1>
        <p className="muted hero-copy">
          Upload one or more files, then ask a question. The backend agent will choose the right tools and return a grounded answer.
        </p>
      </header>
      <UploadAndAsk />
    </main>
  );
}
