from data import load_ndfa, load_brackets
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def padding(x, w2i, batch_size=15):
    batches_x = []

    # step over x met steps of batch_size
    for i in range(0, len(x), batch_size):

        start = i
        end = i + batch_size

        # get the batch
        batch_x = x[start:end]
        batch = []

        # Adding start/end
        for sentence in batch_x:
            sentence.insert(0, w2i['.start'])
            sentence.append(w2i['.end'])

        for i, sentence in enumerate(batch_x):
            longest_sentence = max([len(sentence) for sentence in batch_x])
            if len(sentence) < longest_sentence:
                sentence += [w2i['.pad']] * (longest_sentence - len(sentence))
            # print(len(sentence))
            batch.append(sentence)

        batches_x.append(batch)

    # transform all batches to tensors
    batches_x = [torch.tensor(batch, dtype=torch.long) for batch in batches_x]

    return batches_x

x_train, (i2w, w2i) = load_ndfa(n=150_000)
print(i2w)
print(len(i2w))
batch_x = padding(x_train, w2i)

# Set up the model
class AutoregressiveLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers):
        super(AutoregressiveLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.linear(lstm_out)
        return output

# Instantiate the model
embedding_dim = 32
hidden_size = 16
num_layers = 1
# Give the model
model = AutoregressiveLSTM(len(w2i), embedding_dim, hidden_size, num_layers)

# Initiate hyper-parametric values
learning_rate = 0.001
num_epochs = 10
batch_size = 15

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    total_loss = 0

    # Iterate over batches
    for batch in batch_x:
        optimizer.zero_grad()  # Zero the gradients
        input_batch = batch[:, :-1]  # Input sequence (exclude last token)
        target_batch = batch[:, 1:]  # Target sequence (exclude first token)

        # Forward pass
        output = model(input_batch)

        # Calculate loss
        loss = criterion(output.reshape(-1, len(w2i)), target_batch.reshape(-1))

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Print the average loss for the epoch
    average_loss = total_loss / len(batch_x)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss}')

# Save the trained model (if needed)
torch.save(model.state_dict(), 'autoregressive_lstm_model.pth')