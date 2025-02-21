import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import remarkMath from "remark-math";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw"
import ReactMarkdown from "react-markdown";
import { useState } from "react";
import "/src/pages/MarkdownStyles.css"
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { atomDark } from "react-syntax-highlighter/dist/esm/styles/prism"; // Choose your theme
import { materialLight } from "react-syntax-highlighter/dist/esm/styles/prism";
import { solarizedlight } from "react-syntax-highlighter/dist/esm/styles/prism";

function Home() {
    const markdown = `

### Welcome!

This project started from a growing interest in machine learning and since I was always interested in markets, 
I wanted to see if I could blend the two and gain deeper insights into market behavior using ML.

The journey hasn’t been easy—time series prediction is one of the most challenging tasks in machine learning, especially in a market filled with noise and uncertainty. When I started, I was neither an expert in ML nor in trading, but through trial and error, I gained valuable knowledge along the way. Sometimes, you have to run before you walk!

I hope you find this site informative and useful. If you’d like to discuss or share insights, feel free to reach out to me on Telegram (@frostbitepillars). 🚀

If you wish to contribute please do so by PRs to https://github.com/UdhayaShan1/forex-ml-exploration/tree/main 😀

<img src="${import.meta.env.BASE_URL}images/the-big-short.png" alt="A very important quote" width="900px">
    
    `;


    return <div className="markdown-container">
              <ReactMarkdown
                  remarkPlugins={[remarkMath, remarkGfm]}
                  rehypePlugins={[rehypeKatex, rehypeRaw]}
                  components={{
                      code({ node, inline, className, children, ...props }) {
                          const match = /language-(\w+)/.exec(className || "");
                          return !inline && match ? (
                              <SyntaxHighlighter
                                  style={atomDark}
                                  language={match[1]}
                                  PreTag="div"
                                  {...props}
                              >
                                  {String(children).replace(/\n$/, "")}
                              </SyntaxHighlighter>
                          ) : (
                              <code className={className} {...props}>
                                  {children}
                              </code>
                          );
                      }
                  }}
              >
                  {markdown}
              </ReactMarkdown>
          </div>
}

export default Home;