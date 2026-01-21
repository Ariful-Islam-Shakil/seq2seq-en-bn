class Trainer:
    """
    Docstring for Trainer
    A trainer class to handle the training process of the Seq2Seq model.
    It manages the model, optimizer, and loss criterion.
    #EXAMPLES:
        model = Seq2Seq(encoder, decoder, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
        trainer = Trainer(model, optimizer, criterion)
        src = torch.tensor([[1, 2, 3], [4, 5, 6]])  # Example source batch
        tgt = torch.tensor([[1, 7, 8], [1, 9, 10]])  # Example target batch
        loss = trainer.train_step(src, tgt)
        print(loss)  # Prints the training loss for the step
    """
    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    def train_step(self, src, tgt):
        self.optimizer.zero_grad()
        output = self.model(src, tgt)
        loss = self.criterion(
            output[:, 1:].reshape(-1, output.size(-1)),
            tgt[:, 1:].reshape(-1)
        )
        loss.backward()
        self.optimizer.step()
        return loss.item()