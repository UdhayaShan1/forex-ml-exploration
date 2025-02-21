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
import "/src/pages/LogisticStyles.css";

function LogisticImplementation() {

    const markdown = `
    
# **📌 Logistic Regression Code and Results**

Okay so let's see logistic regression in action and it's results.

The data we will be using is EURUSD Minute 5 bars, from 2023-06-01 to 2023-12-31, provided by Darwinex on their MetaTrader 5 platform.

You can download it here: https://drive.google.com/file/d/1mmgMsvK4aAWDq6Nf9SCk-ap_9aYHxd_O/view?usp=sharing

Also you can access my Colab notebook for this section here: https://colab.research.google.com/drive/1Hz788EdgLwN763EiOwpPs3dtU5xS66Hw?usp=sharing

Okay let's get started! Most of the setup code is similar to the one on the **'Misleading Regression'** page up till the sequence creation.

## 1. Target, lookahead and sequence length

We will do a one day lookahead and our target will be defined as the direction of price change. 

\`\`\`python
#Predict next days closing price
lookahead = 24*12 #24*12*5 = 1440 minutes = 1 day
df['Target'] = np.where(df['Close'].shift(-lookahead) > df['Close'], 1, 0)
df.dropna(inplace=True)
\`\`\`

We will start with sequence length 1000.

\`\`\`python
seq_length = 1000
X_train_seq, Y_train_seq = create_seq(train_scaled, train_target, seq_length)
X_test_seq, Y_test_seq = create_seq(test_scaled, test_target, seq_length)
\`\`\`

\`\`\`python
print("X_train_seq:", X_train_seq.shape)
print("Y_train_seq:", Y_train_seq.shape)
print("X_test_seq:", X_test_seq.shape)
print("Y_test_seq:", Y_test_seq.shape)
\`\`\`

\`X_train_seq: (33728, 1000, 4)
Y_train_seq: (33728,)
X_test_seq: (7683, 1000, 4)
Y_test_seq: (7683,)\`

## 2. Data setup

Currently our X_seq format is $$(batch, seq\\_length, 4)$$, where 4 is the four features of OHLC.
However, logistic regression requires flat (non-sequential) input data. Therefore, we need to reshape our sequential data accordingly, converting it into a format suitable for logistic regression.

That is flatten the $$(seq\\_length, 4)$$ into a single dimension.

\`\`\`python
#X_train_seq.shape[0] refers to batch_size, so .reshape(X_train_seq.shape[0], -1) tells it to keep (batch, flatten rest..)
X_train_reshaped = X_train_seq.reshape(X_train_seq.shape[0], -1)
X_test_reshaped = X_test_seq.reshape(X_test_seq.shape[0], -1)

print("X_train_reshaped:", X_train_reshaped.shape)
print("X_test_reshaped:", X_test_reshaped.shape)
\`\`\`

\`X_train_reshaped: (33728, 4000)
X_test_reshaped: (7683, 4000)\`

We will analyse the distribution of classes, particularly the training class distribution.
\`\`\`python
#View distribution of classes
from collections import Counter
train_distribution = Counter(Y_train_seq)
print("Train Distribution", train_distribution)
print("Training Positive Class %", train_distribution[1]*100/(train_distribution[0] + train_distribution[1]))
print("")
print("Test Distribution", train_distribution)
test_distribution = Counter(Y_test_seq)
print("Test Positive Class %", test_distribution[1]*100/(test_distribution[0] + test_distribution[1]))
\`\`\`

\`Train Distribution Counter({1: 17905, 0: 16813})
Training Positive Class % 51.572671236822394\`

\`Test Distribution Counter({1: 17905, 0: 16813})
Test Positive Class % 52.78450363196126\`

The training classes are rather balanced, so that's good.

## 2. Creating our model

We will utilize \`sklearn\`'s \`LogisticRegression\` to train our model.

Some important parameters of \`LogisticRegression\`
- $$C$$, Inverse of regularization strength

- $$class\\_weight$$, The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data

- $$solver$$, Algorithm to use in the optimization problem. 

Let's create and fit/train our model. We will start with \`class_weight='balanced'\`. \`max_iter\` is set to 10000 in case it takes many iterations to converge.

\`\`\`python
model = LogisticRegression(class_weight='balanced', max_iter=10000)
model.fit(X_train_reshaped, Y_train_seq)
\`\`\`

## 3. Test Results

We will use similar metrics defined in the **'Classification and Metrics'** page.

> 💡 **Recap:**  
> Accuracy measures how often it makes the correct prediction. However, it is is misleading in imbalanced datasets. $$\\\\[1ex]$$
> AUC-ROC evaluates how well a model can distinguish between positive and negative classes. 


\`\`\`python
y_pred = model.predict(X_test_reshaped)
y_pred_probs = model.predict_proba(X_test_reshaped)[:, 1]
print(classification_report(Y_test_seq, y_pred))
print("AUC:", roc_auc_score(Y_test_seq, y_pred_probs))
print("Accuracy", accuracy_score(Y_test_seq, y_pred))

binary_f1 = f1_score(Y_test_seq, y_pred, pos_label=1)
micro_f1 = f1_score(Y_test_seq, y_pred, average='micro')
macro_f1 = f1_score(Y_test_seq, y_pred, average='macro')
weighted_f1 = f1_score(Y_test_seq, y_pred, average='weighted')
print("")
print("Binary F1:", binary_f1)
print("Micro F1:", micro_f1)
print("Macro F1:", macro_f1)
print("Weighted F1:", weighted_f1)
\`\`\`

<figure>
  <img src="${import.meta.env.BASE_URL}images/logistic_impl/500seq.png" alt="Logistic with Seq Length 500" width="500px">
</figure>

**\`AUC: 0.4519349046257069\`**

**\`Accuracy 0.44836856898448\`**

**\`Binary F1: 0.27146546158812135\`**

**\`Micro F1: 0.44836856898448\`**

**\`Macro F1: 0.4138054938225759\`**

**\`Weighted F1: 0.4139272560400562\`**

Since, the classes are rather balanced, we will use \`Accuracy\` and \`AUC\` as our metrics.

- **Accuracy = 0.448**, which is worse than random guessing(50%).
Since the classes are balanced, accuracy is not misleading, but it shows that the model is not learning useful patterns.

- Similarly, **AUC = 0.452** means the model is still worse than random guessing (AUC = 0.5).

<figure>
  <img src="${import.meta.env.BASE_URL}images/logistic_impl/roc-auc-500seq.png" alt="ROC-AUC for 500 Seq" width="500px">
</figure>

**The model is greatly underperforming and not learning any meaningful patterns.**

## 4. Tweaking Parameters 🤔 

### Sequence Length

The model's performance was evaluated using different sequence lengths, and the results are as follows:

| **Sequence Length** | **Accuracy**| **AUC** |
|--------------|------------------| ------ |
| **1**  |  **0.509** | **0.586** | 
| **10**  |  0.509 | 0.582 | 
| **50**  |  0.505 | 0.576 | 
| **100**  |  0.494 | 0.558 | 
| **200**  |  0.462 | 0.478 | 
| **500**  |  0.448 | 0.452 | 

**🔹 Key Observations**
- The model performs best with only the latest day's features (sequence length = 1).

- As the sequence length increases, both accuracy and AUC decline.

- This suggests that recent data is more relevant, while longer historical sequences might introduce noise or reduce the model’s ability to generalize effectively.

- The performance is inversely proportional to sequence length, indicating that longer historical data might not be beneficial for this particular task.

### Grid Search

We can futher tune the hyperparameters for our \`LogisticRegression\` using \`GridSearchCV\`. 

\`GridSearchCV\` automates hyperparameter tuning by searching through a predefined grid of hyperparameters and selecting the best combination based on model performance.

Cross-validation (CV) ensures that the model's performance is not dependent on a single train-test split. It reduces overfitting by training and testing on different subsets of the dataset.

\`\`\`python
from sklearn.model_selection import GridSearchCV

grid = {"C" : [0.001, 0.01, 0.1, 1], 'solver' : ['lbfgs', 'liblinear', 'saga'], 'penalty' : ["l2"]}

grid_search = GridSearchCV(LogisticRegression(class_weight='balanced', max_iter=10000), grid, cv=5)
grid_search.fit(X_train_reshaped, Y_train_seq)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
print(best_params)
\`\`\`

\`{'C': 0.01, 'penalty': 'l2', 'solver': 'liblinear'}\`

<figure>
  <img src="${import.meta.env.BASE_URL}images/logistic_impl/1seq-gridsearch.png" alt="Logistic with Seq Length 1 After Grid Search" width="500px">
</figure>

\`\`\`python
print_metrics(best_model, X_test_reshaped, Y_test_seq)
\`\`\`

**\`Accuracy\`** is 0.509

**\`AUC\`** is 0.586

Since the performance remains nearly unchanged across different settings, applying Grid Search for hyperparameter tuning is unlikely to provide significant improvements.

## 5. Conclusion 😔

The results indicate that Logistic Regression does not perform well for this task.

- The accuracy (0.509) is close to random guessing (0.5), suggesting that the model struggles to learn meaningful patterns.

- The AUC (0.586) shows slight improvement over random guessing but is still far from an effective model.

- Hyperparameter tuning (Grid Search) would likely yield minimal gains, as the core issue appears to be model suitability rather than fine-tuning parameters.

### Next Steps?

**We will further explore non-linear classifiers and see if captures the complex relationships such as Random Forest/XGBoost.**











    
    
    `;

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

export default LogisticImplementation;