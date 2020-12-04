import torch
import torch.nn as nn
import torch.nn.functional as F

# input (32x32x3)
# conv1 (28x28x6) (relu)
# pooling (14x14x6)
# conv2 (10x10x16) (relu)
# pooling (5x5x16)
# fc1(400->100) (relu)
# fc2 (100->80) (relu)
# fc3 (80->10)

class ShallowNetwork(nn.Module):
  def __init__(self):
    super(ShallowNetwork, self).__init__()
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.pool = nn.MaxPool2d(2,2)
    self.fc1 = nn.Linear(400,120)
    self.fc2 = nn.Linear(120,84)
    self.fc3 = nn.Linear(84,14)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 400)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

class GatingNetwork(nn.Module):
  def __init__(self):
    super(GatingNetwork, self).__init__()
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.pool = nn.MaxPool2d(2,2)
    self.fc1 = nn.Linear(400,120)
    self.fc2 = nn.Linear(120,84)
    self.fc3 = nn.Linear(84,4)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 400)
    x = F.leaky_relu(self.fc1(x))
    x = F.leaky_relu(self.fc2(x))
    x = self.fc3(x)
    return x

class GroupOfExpertsShallowNetwork(nn.Module):
  def __init__(self):
    super(GroupOfExpertsShallowNetwork, self).__init__()
    self.challenge_free = ShallowNetwork()
    self.exposure = ShallowNetwork()
    self.codec_error = ShallowNetwork()
    self.lens_blur = ShallowNetwork()
    self.gating_network = GatingNetwork()

  def forward(self, x):
    intermediate = torch.stack([self.challenge_free(x), self.exposure(x), self.codec_error(x), self.lens_blur(x)], 1) # n,4,14
    gate_weights = F.softmax(self.gating_network(x)).reshape(-1,4,1).expand_as(intermediate) # n,4

    x = torch.sum(intermediate * gate_weights, 1)
    return x


class GatingNetwork_all_classes(nn.Module):
  def __init__(self):
    super(GatingNetwork_all_classes, self).__init__()
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.pool = nn.MaxPool2d(2,2)
    self.fc1 = nn.Linear(400,120)
    self.fc2 = nn.Linear(120,84)
    self.fc3 = nn.Linear(84,13)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 400)
    x = F.leaky_relu(self.fc1(x))
    x = F.leaky_relu(self.fc2(x))
    x = self.fc3(x)
    return x


class DeeperCNNRGB_GatingNetwork(nn.Module):
  def __init__(self):
    super(DeeperCNNRGB_GatingNetwork, self).__init__()
    self.conv1 = nn.Conv2d(3, 32, 3, padding=1, padding_mode='reflect')
    self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
    self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
    self.conv4 = nn.Conv2d(64,64, 3, padding=1)
    self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
    self.conv6 = nn.Conv2d(128, 128, 3, padding=1)

    self.bn32 = nn.BatchNorm2d(32)
    self.bn64 = nn.BatchNorm2d(64)
    self.bn128 = nn.BatchNorm2d(128)

    self.pool = nn.MaxPool2d((2,2))
    self.dropout = nn.Dropout2d(.3)

    self.fc1 = nn.Linear(128*4*4, 128)
    self.fc2 = nn.Linear(128, 13)
    self.softmax = nn.Softmax()

  def forward(self, x):
    x = F.leaky_relu(self.conv1(x))
    x = self.bn32(x)
    x = F.leaky_relu(self.conv2(x))
    x = self.pool(x)
    x = self.dropout(x)
    
    x = F.leaky_relu(self.conv3(x))
    x = self.bn64(x)
    x = F.leaky_relu(self.conv4(x))
    x = self.pool(x)
    x = self.dropout(x)

    x = F.leaky_relu(self.conv5(x))
    x = self.bn128(x)
    x = F.leaky_relu(self.conv6(x))
    x = self.pool(x)
    x = self.dropout(x)
    # print(x.shape)
    x = x.view(-1, 128*4*4)
    x = F.leaky_relu(self.fc1(x))
    x = self.dropout(x)
    x = self.fc2(x)
    
    return x

class GroupOfExperts_AllModels_ShallowGating(nn.Module):
  def __init__(self):
    super(GroupOfExperts_AllModels_ShallowGating, self).__init__()
    self.experts = nn.ModuleList([ShallowNetwork() for i in range(13)])
    self.gating_network = GatingNetwork_all_classes()
  def forward(self, x):
    intermediate = torch.stack([expert(x) for expert in self.experts], 1) # n,13,14
    gate_weights = F.softmax(self.gating_network(x)).reshape(-1,13,1).expand_as(intermediate) # n,13

    x = torch.sum(intermediate * gate_weights, 1)
    return x

class GroupOfExperts_AllModels_DeeperGating(nn.Module):
  def __init__(self):
    super(GroupOfExperts_AllModels_DeeperGating, self).__init__()
    self.experts = nn.ModuleList([ShallowNetwork() for i in range(13)])
    self.gating_network = DeeperCNNRGB_GatingNetwork()
  def forward(self, x):
    intermediate = torch.stack([expert(x) for expert in self.experts], 1) # n,13,14
    gate_weights = F.softmax(self.gating_network(x)).reshape(-1,13,1).expand_as(intermediate) # n,13

    x = torch.sum(intermediate * gate_weights, 1)
    return x