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

## Metrics to use

### ✅ 1. Accuracy

**📌 Accuracy** is defined as $$\\frac{TP+TN}{TP+TN+FP+FN}$$

However, accuracy is not useful when classes are imbalanced.

For example, if:
- 90% of the data belongs to class '1' and 10% to class '0'

- The model predicts all instances as '1', accuracy will be 90%, but the model is failing to classify '0' at all.

### ✅ 2. Precision, Recall, and F1-Score

Since accuracy can be misleading, we use Precision, Recall, and F1-score for a better understanding.

**📌 Precision** is defined as $$\\frac{TP}{TP+FP}$$.
- Measures the percentage of positively classified instances that are actually correct.

- Useful when false positives are costly, e.g., predicting a buy signal when the market is falling.

**📌 Recall** is defined as $$\\frac{TP}{TP+FN}$$

- Measures how many actual positives were correctly classified.

- Useful when missing a positive instance is costly, e.g., missing a profitable trade signal.

**📌 F1-Score (Harmonic Mean of Precision & Recall)** is defined as $$2*\\frac{Precision*Recall}{Precision+Recall}$$. 

- Balances precision and recall.

- Best used when both false positives and false negatives are problematic.

### ✅ 3. AUC-ROC (Area Under the Receiver Operating Characteristic Curve)

The ROC Curve is a plot of:
- True Positive Rate (Recall) vs. False Positive Rate (FPR) at different classification thresholds.

- The AUC (Area Under the Curve) quantifies the model's ability to separate classes.

**📌 Why Use AUC-ROC?**

- **Threshold-independent:** Unlike accuracy, it evaluates performance across different probability thresholds.

- **Useful for imbalanced datasets:** Even if one class is more frequent, AUC-ROC remains meaningful.

| **AUC Value** | **Interpretation**     |
|--------------|------------------|
| **0.9 - 1.0**  | Excellent model   |
| **0.8 - 0.9**  | Good model        |
| **0.7 - 0.8**  | Fair model        |
| **0.6 - 0.7**  | Poor model        |
| **0.5**        | Random guessing   |

For example, a AUC-ROC for my Logistic Regression with AUC 0.59.
This shows the model is slightly better than random.

<figure>
  <img src="${import.meta.env.BASE_URL}images/roc-curve-1seq-logistic.png" alt="AUC-ROC for Logistic Regression with 1 seq" width="500px">
</figure>


## Summary of Metrics

| **Metric** | **Use Case**     |
|--------------|------------------|
| **Accuracy**  | Only when classes are balanced.   |
| **Precision**  | When false positives are costly (e.g., incorrect buy signals).        |
| **Recall**  | When false negatives are costly (e.g., missing trading opportunities).     |
| **F1-Score**  | When both false positives and false negatives matter.        |
| **AUC-ROC**        | To measure model’s ranking ability and performance across different thresholds.   |
    
    
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
  );

}

export default Classification;