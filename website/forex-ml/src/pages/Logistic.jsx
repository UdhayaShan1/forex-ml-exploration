import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import remarkMath from "remark-math";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw"
import ReactMarkdown from "react-markdown";
import { useEffect, useState } from "react";
import "/src/pages/MarkdownStyles.css"
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { atomDark } from "react-syntax-highlighter/dist/esm/styles/prism"; // Choose your theme
import { materialLight } from "react-syntax-highlighter/dist/esm/styles/prism";
import { solarizedlight } from "react-syntax-highlighter/dist/esm/styles/prism";

function validNumber(str) {
    let checkStr = str.trim();
    if (isNaN(checkStr) || checkStr.length === 0 || !isFinite(Number(checkStr))) {
        return false;
    }

    let number = Number(checkStr);
    if (number <= 0 || number > 1000) {
        return false;
    }
    return true;
}

function sigmoid2(x) {
    return 1 / (1 + Math.exp(-x));
}

function Logistic() {
    const [features, setFeatures] = useState({x1 : "", x2 : "", x3 : "", x4 : ""});
    const [featuresError, setFeatureError] = useState("");

    const [activate, setActivate] = useState(false);
    const [linearSum , setLinearSum] = useState("");
    const [sigmoid, setSigmoid] = useState("");
    const [threshold, setThreshold] = useState(0.5);
    const [tradeFeedback, setTradeFeedback] = useState("");

    useEffect(() => {
        let stringError = "";
        if (!validNumber(features.x1)) {
            stringError = "Enter a Open Price between 1 and 200";
        } else if (!validNumber(features.x2)) {
            stringError = "Enter a High Price between 1 and 200";
        } else if (!validNumber(features.x3)) {
            stringError = "Enter a Low Price between 1 and 200";
        } else if (!validNumber(features.x4)) {
            stringError = "Enter a Close Price between 1 and 200";
        }

        if (stringError.length != 0) {
            setFeatureError(stringError);
        } else {
            setActivate(true);
            setFeatureError("Submit!");
        }
    }, [features]);

    const onChangeFeatures = (key, value) => {
        setFeatures(prev => ({
            ...prev,
            [key] : value
        }))
    };

    const markdown = `

# **📌 Logistic Regression on Forex**

We will start simple with linear classifiers and work our way to more complex models. 

Logistic regression is a statistical model used to predict binary outcomes (e.g., price going up or down in Forex trading).

### 1. Logistic Regression Formula

Logisitc Regression assumes a linear relationship between the logit and features.

$$ z = \\beta_0 + \\beta_1X_1 + \\beta_2X_2 + ... + \\beta_nX_n$$
Where,
- $$\\beta_0$$ is the bias/intercept term 
- $$\\beta_1, \\beta_2, ... \\beta_n$$ are the coefficients of our model.
- $$X_1, X_2, ..., X_n$$ are the features of our model .



However, since we are interested in classifying instead of predicting a continuous output, we make use of the sigmoid, $$\\sigma(z)$$ function
to squash our output into a probability range between 0 and 1.

That is,
$$\\sigma(z) = \\frac{1}{1+e^{-z}}$$ where $$ z = \\beta_0 + \\beta_1X_1 + \\beta_2X_2 + ... + \\beta_nX_n$$

<figure>
  <img src="${import.meta.env.BASE_URL}images/logistic/sigmoid.png" alt="Sigmoid output" width="500px">
  <figcaption>Source: <a href="https://www.spiceworks.com/tech/artificial-intelligence/articles/what-is-logistic-regression/" target="_blank">https://www.spiceworks.com/tech/artificial-intelligence/articles/what-is-logistic-regression/</a></figcaption>
</figure>

### 2. Decision Boundary

After output, we use a threshold boundary to determine our decision,

$$
\\hat{Y} =
\\begin{cases} 
1, & \\text{if } P(Y=1 | X) \\geq 0.5  \\quad (\\text{Buy Signal}), \\\\[1ex]
0, & \\text{if } P(Y=1 | X) < 0.5  \\quad (\\text{Sell Signal})
\\end{cases}
$$

> 💡 **Note:**  
> However, do note that the boundary is not fixed at 0.5 and can be optimized by cross-validation on training data.  
> While 0.5 usually works well for balanced sets, this may not always be the case.

### 3. Example with Forex Features

In our regression section, we use OHLC (Open, High, Low, Close) prices as our features. For simplicity, we set sequence length as 1.

**Suppose, our relation**

$$ z = \\beta_0 + Open * X_1 + High * X_2 + Low * X_3 + Close * X_4$$

**And a sample of**

$$[1.2841,  1.2932,  1.2735,  1.2774]$$ on 1992-01-02 

**With coefficients** $$\\\\[1ex]$$
$$\\beta_0, \\beta_1, \\beta_2, \\beta_3, \\beta_4 = [0.03522774, -0.88898459,  4.131525,   -4.70063957,  0.34692396]$$

**After subbing in,**

$$ z = -1.3065 $$

**Applying the sigmoid function,**

$$ \\frac{1}{1+e^{-1.3065}} = 0.2131$$

This means that based on the given Forex data and coefficients, the model predicts a 21.31% probability of the price moving up. Since this is below 0.5, it would likely be classified as a Sell Signal. 📉

### 4. Training the model

Unlike linear regression, which may have a closed-form solution, logistic regression relies on the sigmoid function and a threshold to classify outputs.

Since there is no direct solution for finding the optimal coefficients, logistic regression uses an iterative approach to minimize the binary cross-entropy loss function. 
This is achieved through gradient descent, where the model updates its weights (coefficients) and bias step by step to improve predictions.

However, modern libraries like \`sklearn's LogisticRegression\` does most of the heavy lifiting anyways.

### 5. Play around
    `

    const onCalculate = (e) => {
        e.preventDefault();
        setFeatureError("");
        //-0.88898459  4.131525   -4.70063957  0.34692396
        let x1 = Number(features.x1);
        let x2 = Number(features.x2);
        let x3 = Number(features.x3);
        let x4 = Number(features.x4);
        let result = -0.88898459 * x1 + 4.131525 * x2 + -4.70063957 * x3 + 0.34692396 * x4;
        setLinearSum(result.toFixed(3));
        let sigmoidResult = sigmoid2(result);
        console.log(sigmoidResult);
        setSigmoid(sigmoidResult.toFixed(3));

        if (sigmoidResult >= threshold) {
            setTradeFeedback("buy");
        } else {
            setTradeFeedback("sell");
        }
    }

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

          <input type='text' placeholder="Input Open" onChange={(e) => onChangeFeatures("x1", e.target.value)}></input>
          <input type='text' placeholder="Input High" onChange={(e) => onChangeFeatures("x2", e.target.value)}></input>
          <input type='text' placeholder="Input Low" onChange={(e) => onChangeFeatures("x3", e.target.value)}></input>
          <input type='text' placeholder="Input Close" onChange={(e) => onChangeFeatures("x4", e.target.value)}></input>
          {activate && <button onClick={onCalculate}>Calculate</button>}

          {featuresError && <p>{featuresError}</p>}
          {linearSum && <p>Linear Sum is {linearSum}</p>}
          {sigmoid && <p>Sigmoid is {sigmoid}</p>}
          {tradeFeedback && <p>A {tradeFeedback} signal!</p>}
      </div>
  );

}

export default Logistic;