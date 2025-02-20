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

function Classification() {
    const markdown = `
# **📌 Moving from Regression to Classification in Forex Prediction**

After exploring regression-based price prediction, we found that **naive models tend to simply replicate past prices** rather than truly forecasting future movements. While this results in **low Mean Squared Error (MSE)**, it provides **little practical value for trading**.  

To build a **more actionable model**, we shift our focus from predicting exact price values to **predicting the direction of price movement**—a **classification task**. This turns 
out to be a much harder task.

## **Why is Classification More Useful?**  

### _Direction Matters More Than Exact Price_

In trading, knowing whether the price will rise or fall is often more valuable than knowing the exact future price.

A small price difference can still be profitable if the direction is correctly predicted.

### _Avoiding the Pitfall of Overfitting to Past Prices_

In regression, the model minimizes MSE by staying close to past values, leading to predictions that mimic a moving average.

With classification, the model is forced to differentiate between rising and falling markets, leading to a more meaningful prediction.

## **📌 How Do We Define the Classification Task?**  

Recall for our regression task, we set target as the literal price of \`lookahead\` days

\`\`\`python
#Predict next days closing price
lookahead = 1
df['Target'] = df['Close'].shift(-lookahead)
df.dropna(inplace=True)
\`\`\`

Instead, we set it to a binary target of whether closing price of \`lookahead\` days ahead is higher or lower than current.

\`\`\`python
#Predict next days closing price
lookahead = 1
df['Target'] = np.where(df['Close'].shift(-lookahead) > df['Close'], 1, 0)
df.dropna(inplace=True)
\`\`\`




    
    
    `

    return (
      <div className="markdown-container">
          <ReactMarkdown
              remarkPlugins={[remarkMath, remarkGfm]}
              rehypePlugins={[rehypeKatex, rehypeRaw]}
              components={{
                  code({ node, inline, className, children, ...props }) {
                      const match = /language-(\w+)/.exec(className || "");
                      return !inline && match ? (
                          <SyntaxHighlighter
                              style={atomDark} // You can change the theme
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
  );

}

export default Classification;