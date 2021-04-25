from torch import nn

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        
        # input layer
        self.l1 = nn.Linear(5000, 2000) # input shape (5008,) -> output to layer 2000 units
        self.relu1 = nn.ReLU()
        
        # hidden layer 1
        self.l2 = nn.Linear(2000, 500)  # input shape 2000 -> output to layer 1000 units
        self.relu2 = nn.ReLU()
        
        # hidden layer 2
        self.l3 = nn.Linear(500, 100)    # input shape 500 -> output to layer 100 units
        self.relu3 = nn.ReLU()
        
        # hidden layer 3
        self.l4 = nn.Linear(100, 20)    # input shape 100 -> output to layer 20 units
        self.relu4 = nn.ReLU()
        
        # output layer
        self.l5 = nn.Linear(20, 2)      # input shape 20 -> output layer 2 units (binary classifier)
        
    def forward(self, X):
        out = self.l1(X)
        out = self.relu1(out)
        
        out = self.l2(out)
        out = self.relu2(out)
        
        out = self.l3(out)
        out = self.relu3(out)
        
        out = self.l4(out)
        out = self.relu4(out)
        
        out = self.l5(out)
        return out