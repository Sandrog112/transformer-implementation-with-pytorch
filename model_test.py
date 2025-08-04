import sys  
sys.path.append("c:\\Users\\AleksandreKurtishvil\\Desktop\\mastering-nlp-through-paper-implementation")
import torch   
import torch.optim as optim
from torch import nn    
from model.transformer import Transformer

src_vocab_size = 5000
tgt_vocab_size = 5000
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = 100
dropout = 0.1

transformer = Transformer(tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))
tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))


criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

transformer.train()

for epoch in range(10):
    optimizer.zero_grad()
    output = transformer(src_data, tgt_data[:, :-1])
    output = output.reshape(-1, tgt_vocab_size)
    loss = criterion(output, tgt_data[:, 1:].reshape(-1))
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")


    transformer.eval()

val_src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))
val_tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))

with torch.no_grad():

    val_output = transformer(val_src_data, val_tgt_data[:, :-1])
    val_output = val_output.reshape(-1, tgt_vocab_size)
    val_loss = criterion(val_output, val_tgt_data[:, 1:].reshape(-1))
    print(f"Validation Loss: {val_loss.item()}")