import { UploadAndAsk } from "../components/upload-and-ask";

export default function HomePage() {
  return (
    <main>
      <h1>Document Agent</h1>
      <p className="muted">
        Upload one or more files, then ask a question. The backend agent will decide which tools to run and return an answer.
      </p>
      <UploadAndAsk />
    </main>
  );
}
