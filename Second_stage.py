
def transformer_build():
    df = pd.DataFrame(df)

    df['state'] = df['state'].apply(torch.Tensor)
    df['action'] = df['action'].apply(torch.Tensor)
    df['reward'] = df['reward'].apply(torch.Tensor)
    df['next_state'] = df['next_state'].apply(torch.Tensor)
    df['done'] = df['done'].apply(lambda x: torch.tensor(x, dtype=torch.bool))

    # DataFrame -> PyTorch 텐서로 변환 (예제: state와 action 병합)
    tensor_data = df.apply(
        lambda row: torch.cat([row['state'], row['action'], row['next_state'], row['done'], row['reward']]),
        axis=1).tolist()
    tensor_batch = torch.stack(tensor_data)

    inputs = tensor_batch[:, :17]  # state 1 , :17
    labels = torch.cat([tensor_batch[:, 17:23], tensor_batch[:, 23].unsqueeze(1)], axis=1)  # reward shape : 1,1

    # 배치 처리
    batch_size = 1024

    # TensorDataset 생성
    dataset = TensorDataset(inputs, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # hyperparameter
    num_heads = 8
    embedding_dim = 17 * num_heads

    class MultiHeadAttention(nn.Module):
        def __init__(self, num_heads, embedding_dim):
            super().__init__()
            self.embedding_dim = embedding_dim
            self.num_heads = num_heads
            self.head_dim = embedding_dim // num_heads

            self.q = nn.Linear(embedding_dim, embedding_dim)
            self.k = nn.Linear(embedding_dim, embedding_dim)
            self.v = nn.Linear(embedding_dim, embedding_dim)

            self.fc = nn.Linear(embedding_dim, embedding_dim)

        def go(self, x):
            batch_size = x.shape[0]

            q = self.q(x)  # batch_num , sequence_length , embed_size : 16 , 16 , 128
            k = self.k(x)
            v = self.v(x)

            q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,
                                                                                2)  # batch , num_head, 16 , 16*8//8 = 16
            k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

            # Scaled Dot-Product Attention
            attention_score = (q @ k.transpose(3, 2)) / (self.head_dim ** 0.5)
            attention_score = torch.softmax(attention_score, dim=-1)
            attention = torch.matmul(attention_score, v)
            attention = attention.transpose(1, 2).contiguous().view(batch_size, -1,
                                                                    self.embedding_dim)  # batch , 8 , sequence_length , 16*8//8 = 16
            attention = attention.reshape(x.shape[0], x.shape[1], x.shape[2])
            output = self.fc(attention)

            return output  # batch , sequence_length , 16*8

    class Pre_process(nn.Module):
        def __init__(self, num_heads, embedding_dim, l=17, dropout=0.1):
            super().__init__()
            self.embed_size = embedding_dim
            self.embedding = nn.Linear(1, embedding_dim)
            self.l = l
            self.dropout = nn.Dropout(dropout)

        def run(self, input):
            batch_num, seq_length = input.shape
            input = input.unsqueeze(2)
            z = self.embedding(input)

            # 단어 임베딩 + position + 문장 소속(binary)
            out = z
            out = self.dropout(out)
            return out  # out.shape : batch_num , sequence_length , embed_size

    #Encoder block
    class Encoder_Block(nn.Module):
        def __init__(self, num_heads, embed_size, dropout=0.1):
            super().__init__()

            self.relu = nn.ReLU()

            self.norm1 = nn.LayerNorm(embed_size)
            self.norm2 = nn.LayerNorm(7)

            self.l1 = nn.Linear(embed_size, embed_size)
            self.l2 = nn.Linear(embed_size, 1)
            self.l3 = nn.Linear(17, 7)

            self.pre_process = Pre_process(num_heads, embed_size)
            self.attention = MultiHeadAttention(num_heads, embed_size)

            self.dropout = nn.Dropout(0.1)

        def update(self, input):
            c = self.pre_process.run(input)
            z = self.attention.go(c)

            z = z + c
            # print(z.shape) , [16, 16, 128]
            z1 = self.norm1(z)

            z = self.l1(z1)
            z = self.relu(z)

            # skip
            z = z1 + z  # z.shape : 16 ,16 , 128

            z = self.l2(z)  # 16 ,16 , 1
            z = z.reshape(z.shape[0], z.shape[1])
            z = self.l3(z)  # 16 , 1

            z = self.norm2(z)  # 16 , 1

            return z  # batch_num , 7 : action [:6] , reward [6]

    #model 설계
    model = Encoder_Block(num_heads, embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    cnt = 0
    loss_history = []
    mse_loss = nn.MSELoss()

    epochs = 10

    model.train()
    for epoch in range(epochs):
        for data, label in dataloader:
            output = model.update(data)

            loss = mse_loss(output, label.unsqueeze(1)).mean()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # loss history append
            loss_history.append(loss.item())

        # scheduler_linear.step()
        print("epoch : {} , loss : {}".format(epoch, loss))


