// File: src/Recommend.jsx

import { useChat } from "ai/react";

export default function () {
  const { messages, handleSubmit, input, handleInputChange } = useChat({
    api: "http://localhost:8000/api/chat",
  });
  return (
    <form
      onSubmit={handleSubmit}
      className="mt-12 flex w-full max-w-[300px] flex-col"
    >
      <input
        id="input"
        name="prompt"
        value={input}
        autoComplete="off"
        onChange={handleInputChange}
        placeholder="Describe the product you're looking for."
        className="mt-3 min-w-[300px] rounded border px-2 py-1 outline-none focus:border-black"
      />
      <button
        type="submit"
        className="mt-3 max-w-max rounded border px-3 py-1 outline-none hover:bg-black hover:text-white"
      >
        Find &rarr;
      </button>
      {messages.map((message, i) =>
        message.role === "assistant" ? (
          <div className="mt-3 border-t pt-3 flex flex-col">
            {
              JSON.parse(message.content).map((product, _) => (
                <div key={product.metadata.image} className="mt-3 flex flex-col">
                    <img alt={product.metadata.title} src={product.metadata.image} />
                    <span>{product.metadata.title}</span>
                    <span>{product.metadata.description}</span>
                </div>
              ))}
          </div>
        ) : (
          <div className="mt-3 border-t pt-3" key={i}>
            {message.content}
          </div>
        )
      )}
    </form>
  );
}