import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F




# ------------------------------------
#              NE Encoder
# ------------------------------------

class ConvNetwork_cifar(torch.nn.Module):
    def __init__(self, input_shape):
        super(ConvNetwork_cifar, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)

        # Calculate the size of the flattened features after the last conv layer
        self.flatten_size = 128 * (input_shape[1] // 8) * (input_shape[2] // 8)

        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x




# class FCNetwork_Cifar(torch.nn.Module):
#     "Fully-connected network"
#     def __init__(self, in_dim=3072, feat_dim=2):
#         super(FCNetwork_Cifar, self).__init__()
#         self.flatten = torch.nn.Flatten()
#         self.linear_relu_stack = torch.nn.Sequential(
#             torch.nn.Linear(in_dim, 1024),
#             torch.nn.ReLU(),
#             torch.nn.Linear(1024, 512),
#             torch.nn.ReLU(),
#             torch.nn.Linear(512, 512),
#             torch.nn.ReLU(),
#             torch.nn.Linear(512, 256),
#             torch.nn.ReLU(),
#             torch.nn.Linear(256, feat_dim),
#         )

#     def forward(self, x):
#         x = self.flatten(x)
#         logits = self.linear_relu_stack(x)
#         return logits




class FCNetwork_Cifar(torch.nn.Module):
    "Fully-connected network"
    def __init__(self, in_dim=512, feat_dim=2):
        super(FCNetwork_Cifar, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, feat_dim),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits



class FCNetwork_mnist_(torch.nn.Module):
    "Fully-connected network"
    def __init__(self, in_dim=784, feat_dim=2):
        super(FCNetwork_mnist_, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, feat_dim),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits



    
    
class FCNetwork_mnist(torch.nn.Module):
    "Fully-connected network"
    def __init__(self, in_dim=784, feat_dim=2):
        super(FCNetwork_mnist, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, feat_dim),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits





class FCNetwork_fashmnist(torch.nn.Module):
    "Fully-connected network"
    def __init__(self, in_dim=784, feat_dim=2):
        super(FCNetwork_fashmnist, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, feat_dim),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits





class FCNetwork_rnaseq(torch.nn.Module):
    "Fully-connected network"
    def __init__(self, in_dim=50, feat_dim=2):
        super(FCNetwork_rnaseq, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, feat_dim),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits




# ------------------------------------
#       ClientFuncRegressionModel
# ------------------------------------
class ClientFuncRegressionModel_rep(nn.Module):
    def __init__(self, e_dim, output_dim):
        super(ClientFuncRegressionModel_rep, self).__init__()

        # self.fc1 = nn.Linear(e_dim, 32)
        # self.fc2 = nn.Linear(32, 128)
        # self.out = nn.Linear(128, output_dim)
        self.fc1 = nn.Linear(e_dim, 100)
        self.fc2 = nn.Linear(100, 100)
        self.out = nn.Linear(100, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.out(x)
        return x






class ClientFuncRegressionModel_att(nn.Module):
    def __init__(self, x_dim, e_dim, output_dim):
        super(ClientFuncRegressionModel_att, self).__init__()

        # For the first input
        self.fc1_1 = nn.Linear(x_dim, 512)
        self.fc1_2 = nn.Linear(512, 256)

        # For the second input
        self.fc2_1 = nn.Linear(e_dim, 16)
        self.fc2_2 = nn.Linear(16, 16)

        # Merging layers
        self.fc3 = nn.Linear(256 + 16, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, output_dim)

    def forward(self, x1, x2):
        # First input
        x1 = F.relu(self.fc1_1(x1))
        x1 = F.relu(self.fc1_2(x1))

        # Second input
        x2 = F.relu(self.fc2_1(x2))
        x2 = F.relu(self.fc2_2(x2))

        # Merge
        x = torch.cat((x1, x2), dim=1)

        # Further processing
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)  # No activation since it's a regression output

        return x



class CustomDataset(Dataset):
    def __init__(self, input, labels):
        self.input = input
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.input[idx], self.labels[idx]



class CustomDataset_xey(Dataset):
    def __init__(self, input1, input2, labels):
        self.input1 = input1
        self.input2 = input2
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.input1[idx], self.input2[idx], self.labels[idx]






class ClientFuncRegressionModel_att_v2(nn.Module):
    def __init__(self, x_dim, e_dim, output_dim):
        super(ClientFuncRegressionModel_att_v2, self).__init__()

        self.fc1 = nn.Linear(e_dim, 512)#128)
        # self.fc2 = nn.Linear(x_dim + 128, 512)
        self.fc2 = nn.Linear(x_dim + 512, 512)

        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, output_dim)

    def forward(self, x1, x2):

        x2 = F.relu(self.fc1(x2))

        x = torch.cat((x1, x2), dim=1)

        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)

        return x



