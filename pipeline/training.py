import torch
from torch.utils.data import DataLoader
from model.transformer import Transformer
from pipeline.data_processing import RandomSequenceDataset, collate_fn

# Hyperparameters
VOCAB_SIZE = 20
SEQ_LENGTH = 10
BATCH_SIZE = 4
DATASET_SIZE = 100
D_MODEL = 32
NUM_HEADS = 4
NUM_LAYERS = 2
D_FF = 64
MAX_SEQ_LENGTH = SEQ_LENGTH
DROPOUT = 0.1
NUM_EPOCHS = 3
LEARNING_RATE = 1e-3

# Dataset and loader
dataset = RandomSequenceDataset(VOCAB_SIZE, SEQ_LENGTH, DATASET_SIZE)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# Model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer(VOCAB_SIZE, D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF, MAX_SEQ_LENGTH, DROPOUT).to(device)
criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for src, tgt in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)
        
        # Prepare inputs and targets for decoder:
        # Usually, decoder input excludes last token, target excludes first token
        decoder_input = tgt[:, :-1]
        target = tgt[:, 1:]

        optimizer.zero_grad()
        output = model(src, decoder_input)  # shape: (batch_size, seq_len-1, vocab_size)
        
        # Reshape output and target for loss calculation
        output = output.reshape(-1, VOCAB_SIZE)
        target = target.reshape(-1)
        
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
