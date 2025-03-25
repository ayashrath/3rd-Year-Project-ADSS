import torch.nn as nn
import torch.optim as optim
from toolkit_dn import Dataset, ModelTrainerDNN


dataset = Dataset()  # add theshold if you need
scalar_label = dataset.clean(scalar_type="standard", save_scalar_val=False)["scalar_label"]
train_loader, test_loader, inp_dim = dataset.return_tensor(batch_size=32)


# Definition of the current Regression DNN
class RegressionNN(nn.Module):
    def __init__(self, input_dim):
        super(RegressionNN, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
        )

        self.regressor = nn.Linear(64, 1)

        for layer in self.feature_extractor:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.regressor(x)


loss_lst = []

# all
model = RegressionNN(inp_dim)
epoch = 1
lr = 0.001
criteria = nn.HuberLoss(delta=1.0)
optimiser = optim.Adam(model.parameters(), lr=lr)
trainer = ModelTrainerDNN(
    model, train_loader, test_loader, scalar_label, criteria, optimiser, epoch, auto_save_model=False
)
loss_lst += trainer.train_model()
trainer.validate(loss_lst, simple=True, train=True)
trainer.validate(loss_lst, simple=True, save_onnx=False)
