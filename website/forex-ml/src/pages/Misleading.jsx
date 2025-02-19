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

function Misleading() {
    const markdown = `

# 📢 Too Good to Be True?

We have all seen LSTM-generated graphs like the one below, showcasing incredible prediction accuracy.

<figure>
  <img src="${import.meta.env.BASE_URL}images/lstm-stock.png" alt="Misleading Result of LSTM" width="500px">
  <figcaption>Source: <a href="https://www.linkedin.com/pulse/predicting-nordeas-stock-price-using-lstm-neural-network-jan-nordin/" target="_blank">https://www.linkedin.com/pulse/predicting-nordeas-stock-price-using-lstm-neural-network-jan-nordin/</a></figcaption>
</figure>

It certainly made me excited about the prospect of using machine learning to trade since it looked too <em>perfect</em>.

So, I decided to replicate this result using PyTorch to predict EUR/USD daily prices. The results were surprisingly good-looking, but are they actually useful?

You can run the same code on my Colab notebook:
https://colab.research.google.com/drive/1RG77M8SvLvx_9CtRQ5qykeKTZpCFk6r9?usp=sharing

## Data Setup

For this demonstration, I will be using Darwinex's EURUSD data from 1992 to 2024. We will rename columns for readability.

\`\`\`python
df = pd.read_csv("eurusd_1992.csv", sep="\\t")

df = df.rename(columns={
    '<DATE>': 'Date',
    '<OPEN>': 'Open',
    '<HIGH>': 'High',
    '<LOW>': 'Low',
    '<CLOSE>': 'Close',
    '<TICKVOL>': 'TickVol',
    '<VOL>': 'Volume',
    '<SPREAD>': 'Spread'
});
\`\`\`

We will drop irrelavant columns and ensure columns are numeric. We will also ensure \`Date\` column if converted to date format and set as index column.

\`\`\`python
# To keep it simple, keep track of date and OHLC
df = df.drop(columns=['Spread', 'TickVol', 'Volume'], errors='ignore')

#Ensure numeric rows are numeric
numeric_cols = ['Open', 'High', 'Low', 'Close']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
df.dropna(subset=numeric_cols, inplace=True)

#Ensure Date column is date format
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.dropna(inplace=True)
\`\`\`

The feature columns that we will be using is simply the OHLC.

\`\`\`python
feature_columns = ['Open', 'High', 'Low', 'Close']
df = df[feature_columns].dropna()
\`\`\`

## 📌 Target Variable
We will use a lookahead of 1 day.

\`\`\`python
#Predict next days closing price
lookahead = 1
df['Target'] = df['Close'].shift(-lookahead)
df.dropna(inplace=True)
\`\`\`

## 🚨 Avoiding a Common Mistake: Proper Time-Series Split

Another mistake is to use \`train_test_split()\` on time series data. This may lead to data leakage, where we train on
future prices to predict earlier prices. Accuracy will be misleadingly high.

We will split by time range, so we will train on the first 80% of the data and test on 20% of the remaining data.

\`\`\`python
#We do a 80-20 split
split_ratio = 0.8
split_index = int(len(df) * split_ratio)

train_data = df.iloc[:split_index]
test_data = df.iloc[split_index:]
\`\`\`

## 📌 Feature Scaling

Normalizing the data will speed up convergence for our training. 

❗However, it is important to scale the input and output separately to prevent data leakage.

\`\`\`python
#Use StandardScaler
scaler = StandardScaler()
price_columns = ['Close', 'High', 'Low', 'Open']
train_scaled = scaler.fit_transform(train_data[price_columns])
test_scaled = scaler.transform(test_data[price_columns])

target_scaler = StandardScaler()
train_target_scaled = target_scaler.fit_transform(train_data[['Target']])[:, 0]
test_target_scaled = target_scaler.transform(test_data[['Target']])[:, 0]
\`\`\`

## Setup up sequences

We setup a sequence length of 100.
\`\`\`python
def create_seq(data, targets, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        # For example: Seq length 4 -> Days 0 to 3 predicts Day 4's closing price
        # Target is defined as next day's closing price, hence target[day 3] -> day 4's close price
        # Hence data[i-seq_length:i] -> days 0 to 3, targets[i-1] -> day 4's close price
        X.append(data[i-seq_length:i])
        y.append(targets[i-1])
    return np.array(X), np.array(y)

seq_length = 1
X_train_seq, Y_train_seq = create_seq(train_scaled, train_target_scaled, seq_length)
X_test_seq, Y_test_seq = create_seq(test_scaled, test_target_scaled, seq_length)

print("X_train_seq:", X_train_seq.shape)
print("Y_train_seq:", Y_train_seq.shape)
print("X_test_seq:", X_test_seq.shape)
print("Y_test_seq:", Y_test_seq.shape)
\`\`\`

\`X_train_seq: (6862, 100, 4)
Y_train_seq: (6862,)
X_test_seq: (1715, 100, 4)
Y_test_seq: (1715,)\`

We can intepret this as, each input comprises of 100 days with each day having 4 features of OHLC. For each of this sequence, we predict next day's price.

## Setup dataset

We convert the input to torch tensors and setup the dataset and dataloader.

\`\`\`python
X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train_seq, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test_seq, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
\`\`\`

## Setup model

We setup a feed forward neural network with two hidden layers. Since we predicting prices, we output 1 feature.

\`\`\`python
class FeedForwardNN(nn.Module):
  def __init__(self, seq_length, num_features):
    super(FeedForwardNN, self).__init__()
    input_dim = seq_length * num_features
    self.layers = nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )
  def forward(self, x):
    return self.layers(x).squeeze(-1)
\`\`\`

## Train the model

\`\`\`python
learning_rate = 1e-5
model = FeedForwardNN(seq_length, len(price_columns))
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True, min_lr=1e-50)

num_epochs = 100
patience = 10
best_val_loss = np.inf
epochs_no_improve = 0

train_losses = []
val_losses = []
best_model_state = None
for i in range(num_epochs):
  model.train()
  closs = 0
  for input, output in train_dataloader:
    input = input.to(device)
    output = output.to(device)
    optimizer.zero_grad()
    prediction = model(input)
    loss = criterion(prediction, output)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    closs += loss.item() * input.size(0)
  closs /= len(train_dataloader.dataset)
  train_losses.append(closs)

  model.eval()
  closs = 0.0
  with torch.no_grad():
      for input, output in test_dataloader:
          input = input.to(device)
          output = output.to(device)
          y_pred = model(input)
          loss = criterion(y_pred, output)
          closs += loss.item() * input.size(0)
  closs = closs / len(test_dataloader.dataset)
  val_losses.append(closs)

  scheduler.step(closs)

  print(f"Epoch {i+1}/{num_epochs} - Train Loss: {train_losses[-1]:.6f} - Val Loss: {val_losses[-1]:.6f}")
  if closs < best_val_loss:
    best_val_loss = closs
    epochs_no_improve = 0
    best_model_state = model.state_dict()
  else:
    epochs_no_improve += 1

  if epochs_no_improve == patience:
    print("Early stopping triggered")
    break
\`\`\`

<figure>
  <img src="${import.meta.env.BASE_URL}images/neural-mse-loss.png" alt="Neural network training and test loss" width="1000px">
</figure>


Pretty good! Converges in less than 20 epochs to near zero. The prediction must look pretty good right?

## Predict and Plot

\`\`\`python
model.eval()
all_preds = []
all_actuals = []
with torch.no_grad():
    for input, output in test_dataloader:
        input = input.to(device)
        output = output.to(device)
        y_pred = model(input)
        all_preds.extend(y_pred.cpu().numpy())
        all_actuals.extend(output.cpu().numpy())

mse = mean_squared_error(all_actuals, all_preds)
mae = mean_absolute_error(all_actuals, all_preds)
print("\\nTest Set Performance:")
print(f"MSE: {mse:.3f}")
print(f"MAE: {mae:.3f}")
\`\`\`

\`Test Set Performance:
MSE: 0.001
MAE: 0.025\`

Remember to inverse transform the predictions as they were scaled earlier.

\`\`\`python
all_preds_scaled = np.array(all_preds)
all_actuals_scaled = np.array(all_actuals)

all_preds = target_scaler.inverse_transform(all_preds_scaled.reshape(-1, 1)).flatten()
all_actuals = target_scaler.inverse_transform(all_actuals_scaled.reshape(-1, 1)).flatten()
\`\`\`

Finally let's plot.

\`\`\`python
plt.figure(figsize=(12, 6))
plt.plot(all_actuals, label='Actual Future Price', color='blue')
plt.plot(all_preds, label='Predicted Future Price', color='red', alpha=0.7)
plt.title('Predicted vs Actual Future Price')
plt.xlabel('Test Sample Index')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
\`\`\`

<figure>
  <img src="${import.meta.env.BASE_URL}images/predicted-actual-price.png" alt="Predicted and Actual Price" width="1000px">
</figure>

**WOW** That's amazing! 

RICHHH, MILIONIAR.... BOOOOOMBOCLAAAAT 🗣️💥🔥💥🗣️💥
https://www.youtube.com/shorts/GFV4RiY1bcI


## 📌 But Are the Predictions Actually Useful?

Well, let's plot a Simple Moving Average with rolling window size 50. This can be done on any trading software and is one of the most basic indicators. It basically measures the average of prices over a window. 

\`\`\`python
# Define moving average window
window_size = 50

# Compute Simple Moving Average (SMA) of actual prices
sma_actuals = pd.Series(all_actuals).rolling(window=window_size).mean()

# Plot actual prices and the moving average
plt.figure(figsize=(12, 6))
plt.plot(all_actuals, label='Actual Future Price', color='blue', alpha=0.6)
plt.plot(sma_actuals, label=f'{window_size}-Step Moving Average', color='red', linestyle='dashed')

plt.title('Actual Price with Moving Average)')
plt.xlabel('Test Sample Index')
plt.ylabel('Price')
plt.legend()
plt.grid(True)

plt.show()
\`\`\`

<figure>
  <img src="${import.meta.env.BASE_URL}images/50sma.png" alt="Actual Price with 50sma" width="1000px">
</figure>

Ok so what? Our model still does better? Let's see what a SMA with window size 1 looks like.

## ⚠️ The Pitfall: Our Model Just Mimics a Moving Average

<figure>
  <img src="${import.meta.env.BASE_URL}images/1sma.png" alt="Actual Price with 1sma" width="1000px">
</figure>

That is exactly our neural network prediction! So what is going on? Let's take a closer look at past 40 days prediction

\`\`\`python
plt.figure(figsize=(12, 6))
plt.plot(all_actuals[-40:], label='Actual Future Price', color='blue')
plt.plot(all_preds[-40:], label='Predicted Future Price', color='red', alpha=0.7)
plt.title('Predicted vs Actual Future Price')
plt.xlabel('Test Sample Index')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
\`\`\`

<figure>
  <img src="${import.meta.env.BASE_URL}images/past40days.png" alt="Past 40 days" width="1000px">
</figure>

Essentially, our neural network is **not truly forecasting** future prices—it is merely **replicating past prices**! By staying close to the previous day's price, regardless of whether the market moves up or down, the model **minimizes its Mean Squared Error (MSE)** and produces what appears to be **highly accurate predictions**.

However, this accuracy is **misleading**. The model does not capture **actual price movements** or **market trends**—it simply smooths out past data. In other words, it is indifferent to **directional changes**, making it **useless for real-world trading**.

Using such a model for trading would be a **bad idea**, as it provides **no real predictive power**—only an illusion of accuracy through historical replication.










    
    
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

};

export default Misleading;