export type Message = {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  modelName?: string;
};

export type Citation = {
  id: string;
  filename: string;
  url?: string;
  source_path?: string;
  metadata: any;
};

export async function chat(
  question: string,
  modelType: "ollama" | "openai",
  modelName: string | undefined,
  onToken: (token: string) => void,
  onCitations: (citations: Citation[]) => void
) {
  const response = await fetch("http://localhost:8005/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      model_type: modelType,
      model_name: modelName,
    }),
  });

  if (!response.ok) throw new Error("Failed to fetch response");
  if (!response.body) return;

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const json = JSON.parse(line);
        if (json.type === "citations") {
          onCitations(json.data);
        } else if (json.type === "token") {
          onToken(json.data);
        }
      } catch (e) {
        console.error("Error parsing line", line, e);
      }
    }
  }
}
