import { useState, useRef, useEffect } from "react";
import { chat, Message } from "./api";
import { Send, BookOpen, Bot, User, Settings2 } from "lucide-react";

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [modelType, setModelType] = useState<"ollama" | "openai">("ollama");
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMsg: Message = { role: "user", content: input };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    const botMsg: Message = {
      role: "assistant",
      content: "",
      modelName: modelType,
    };
    setMessages((prev) => [...prev, botMsg]);

    try {
      await chat(
        userMsg.content,
        modelType,
        undefined,
        (token) => {
          setMessages((prev) => {
            const newMsgs = [...prev];
            const last = newMsgs[newMsgs.length - 1];
            last.content += token;
            return newMsgs;
          });
        },
        (citations) => {
          setMessages((prev) => {
            const newMsgs = [...prev];
            const last = newMsgs[newMsgs.length - 1];
            last.citations = citations;
            return newMsgs;
          });
        }
      );
    } catch (e) {
      console.error(e);
      setMessages((prev) => {
        const newMsgs = [...prev];
        const last = newMsgs[newMsgs.length - 1];
        last.content += "\n[שגיאה בתקשורת]";
        return newMsgs;
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      className="flex flex-col h-screen overflow-hidden bg-gray-50 font-sans"
      dir="rtl"
    >
      {/* Header */}
      <header className="bg-white shadow-sm p-4 flex items-center justify-center sticky top-0 z-10">
        <div className="flex items-center gap-2 text-blue-700">
          <BookOpen className="w-6 h-6" />
          <h1 className="text-xl font-bold text-center w-full">
            מערכת חיפוש משפטי
          </h1>
        </div>
      </header>

      {/* Input Area - Moved to Top */}
      <div className="bg-white p-4 border-b border-gray-200 shadow-md z-20">
        <div className="max-w-4xl mx-auto">
          <form
            onSubmit={handleSubmit}
            className="relative flex flex-col gap-3"
          >
            {/* Model Selector - Pills style */}
            <div className="flex items-center gap-2 self-start text-xs text-gray-500 bg-gray-100 rounded-full px-3 py-1 border border-gray-200 hover:bg-gray-200 transition-colors">
              <Settings2 size={14} />
              <span>מודל פעיל:</span>
              <select
                value={modelType}
                onChange={(e) => setModelType(e.target.value as any)}
                className="bg-transparent font-semibold text-gray-700 focus:outline-none cursor-pointer"
              >
                <option value="ollama">Llama 3 (מקומי)</option>
                <option value="openai">GPT-4 (ענן)</option>
              </select>
            </div>

            <div className="relative flex items-end gap-2 w-full">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="שאל שאלה משפטית בנוגע לפסקי הדין..."
                className="w-full p-4 pl-12 rounded-xl bg-gray-50 border-gray-200 border focus:bg-white focus:border-blue-500 focus:ring-2 focus:ring-blue-200 transition-all outline-none resize-none shadow-sm"
                disabled={loading}
                dir="auto"
              />

              <button
                type="submit"
                disabled={loading || !input.trim()}
                className="absolute left-2 bottom-2 p-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-sm flex items-center justify-center"
              >
                <Send
                  size={20}
                  className={loading ? "animate-pulse" : "transform rotate-180"}
                />
              </button>
            </div>

            <div className="text-center text-xs text-gray-400 mt-1">
              המערכת מתבססת על מסמכים משפטיים ועשויה לטעות. יש לבדוק את המקורות
              המצורפים.
            </div>
          </form>
        </div>
      </div>

      {/* Messages Area - Below Input */}
      <div className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-8 scroll-smooth max-h-full bg-gray-50">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-gray-400 space-y-4">
            <Bot className="w-16 h-16 opacity-20" />
            <p className="text-lg">התוצאות יופיעו כאן...</p>
          </div>
        )}

        {/* Group messages into Interaction Pairs */}
        {messages
          .reduce((acc: { user: Message; bot?: Message }[], msg) => {
            if (msg.role === "user") {
              acc.push({ user: msg });
            } else if (msg.role === "assistant" && acc.length > 0) {
              acc[acc.length - 1].bot = msg;
            }
            return acc;
          }, [])
          .map((interaction, i) => (
            <div
              key={i}
              className="bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden"
            >
              {/* User Question Section */}
              <div className="bg-blue-50 p-6 border-b border-blue-100 flex gap-4 items-start">
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-600 text-white flex items-center justify-center mt-1">
                  <User size={18} />
                </div>
                <div className="flex-1 text-gray-800 font-medium leading-relaxed">
                  {interaction.user.content}
                </div>
              </div>

              {/* Bot Answer Section */}
              {interaction.bot && (
                <div className="p-6 flex gap-4 items-start bg-white">
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-emerald-600 text-white flex items-center justify-center mt-1">
                    <Bot size={18} />
                  </div>
                  <div className="flex-1 text-blue-800 leading-relaxed whitespace-pre-wrap">
                    {interaction.bot.modelName && (
                      <div className="text-xs text-blue-400 mb-2">
                        מודל:{" "}
                        {interaction.bot.modelName === "ollama"
                          ? "Llama 3 (מקומי)"
                          : "GPT-4 (ענן)"}
                      </div>
                    )}
                    {interaction.bot.content}

                    {interaction.bot.citations &&
                      interaction.bot.citations.length > 0 && (
                        <div className="mt-6 pt-4 border-t border-gray-100">
                          <div className="text-xs font-semibold text-gray-500 mb-3 flex items-center gap-2">
                            <BookOpen size={14} />
                            מקורות והפניות:
                          </div>
                          <div className="grid gap-2 sm:grid-cols-2">
                            {interaction.bot.citations.map((cit, idx) => (
                              <div
                                key={idx}
                                className="text-xs text-gray-600 flex gap-2 bg-gray-50 p-2 rounded border border-gray-100 hover:bg-blue-50 hover:border-blue-100 transition-colors"
                              >
                                <span className="font-bold text-blue-500 shrink-0">
                                  {cit.id}
                                </span>
                                {cit.url ? (
                                  <a
                                    href={cit.url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="truncate text-blue-600 hover:text-blue-800 hover:underline cursor-pointer"
                                    title={`פתח קובץ: ${cit.filename}`}
                                  >
                                    {cit.filename}
                                  </a>
                                ) : (
                                  <span
                                    className="truncate"
                                    title={cit.filename}
                                  >
                                    {cit.filename}
                                  </span>
                                )}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                  </div>
                </div>
              )}
            </div>
          ))}
        <div ref={messagesEndRef} />
      </div>
    </div>
  );
}

export default App;
