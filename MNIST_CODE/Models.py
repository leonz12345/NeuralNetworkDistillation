import torch
import torch.nn as nn
import torch.nn.functional as F


class TeacherSimpleNN(nn.Module):
    def __init__(self):
        super(TeacherSimpleNN, self).__init__()
        self.linear1 = nn.Linear(28 * 28, 1200)
        self.linear2 = nn.Linear(1200, 1200)
        self.linear3 = nn.Linear(1200, 10)
        self.dropout_rate = 0.3
        
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        out = self.dropout(x)

        out = self.linear1(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.linear2(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.linear3(out)

        return out


class StudentSimpleNN(nn.Module):
    def __init__(self):
        super(StudentSimpleNN, self).__init__()
        self.linear1 = nn.Linear(28 * 28, 400)
        self.linear2 = nn.Linear(400, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)

        out = self.linear1(x)
        out = F.relu(out)

        out = self.linear2(out)

        return out