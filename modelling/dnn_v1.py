import torch.nn as nn
import torch.optim as optim
from toolkit_dn import Dataset, ModelTrainerDNN


dataset = Dataset()  # add theshold if you need
dataset.clean()
train_loader, test_loader, inp_dim, scalar_label = dataset.format_data("standard", save_scalar_val=True, batch_size=32)


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


"""
DNN 0 - Regression Model on tt between 2nd Percentile and 98th Pencentile
DNN 1 - Classification Model which predicts if an aircraft has tt < 5000 +- 100
DNN 2 - Regression Model trained on tt > 2nd Percentile and tt < 5000 Percentile
DNN 3 - Regression Model trained on tt > 5000 and tt < 98th Percentile

loss_lst_dnn_0 = []
loss_lst_dnn1 = []
loss_lst_dnn2 = []
loss_lst_dnn3 = []
"""

loss_lst = []

# all
model = RegressionNN(inp_dim)
epoch = 100
lr = 0.001
criteria = nn.HuberLoss(delta=1.0)
optimiser = optim.Adam(model.parameters(), lr=lr)
trainer = ModelTrainerDNN(model, train_loader, test_loader, scalar_label, criteria, optimiser, epoch)
loss_lst += trainer.train_model()
trainer.validate(loss_lst, simple=False, train=True)
trainer.validate(loss_lst, simple=False)

# trainer.save_onnx()
