import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

class my3DUnet(BaseModel): 
    def __init__(self, n_features = 64, n_classes = 1, image_size = 64, dropout_rate=0.05):
        super().__init__()
        
        self.features = n_features
        self.in_channels = 1
        self.n_classes = 2             
        self.image_size = image_size
        self.dropout_rate = dropout_rate
        
        self.conv1a = nn.Conv3d(self.in_channels, self.features, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.conv1b = nn.Conv3d(self.features, self.features, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.conv2a = nn.Conv3d(self.features, self.features * 2, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.conv2b = nn.Conv3d(self.features * 2, self.features * 2, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.conv3a = nn.Conv3d(self.features * 2, self.features * 4, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.conv3b = nn.Conv3d(self.features * 4, self.features * 4, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.conv4a = nn.Conv3d(self.features * 4, self.features * 8, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.conv4b = nn.Conv3d(self.features * 8, self.features * 8, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.conv5a = nn.Conv3d(self.features * 8, self.features * 16, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.conv5b = nn.Conv3d(self.features * 16, self.features * 16, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        
        self.up6a = nn.ConvTranspose3d(self.features * 16, self.features * 8, kernel_size=2, stride=2, padding=0)
        self.conv7a = nn.Conv3d(self.features * 16, self.features * 8, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.conv7b = nn.Conv3d(self.features * 8, self.features * 8, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.up8a = nn.ConvTranspose3d(self.features * 8, self.features * 4, kernel_size=2, stride=2, padding=0)
        self.conv9a = nn.Conv3d(self.features * 8, self.features * 4, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.conv9b = nn.Conv3d(self.features * 4, self.features * 4, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.up10a = nn.ConvTranspose3d(self.features * 4, self.features * 2, kernel_size=2, stride=2, padding=0)
        self.conv11a = nn.Conv3d(self.features * 4, self.features * 2, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.conv11b = nn.Conv3d(self.features * 2, self.features * 2, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.up12a = nn.ConvTranspose3d(self.features * 2, self.features, kernel_size=2, stride=2, padding=0)
        self.conv13a = nn.Conv3d(self.features * 2, self.features, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.conv13b = nn.Conv3d(self.features, self.features, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        
        self.conv_out = nn.Conv3d(self.features, self.n_classes, kernel_size=1, stride=1, padding=0)
        
        self.drop1 = nn.Dropout3d(p=dropout_rate)
        self.drop2 = nn.Dropout3d(p=dropout_rate)
        self.drop3 = nn.Dropout3d(p=dropout_rate)
        self.drop4 = nn.Dropout3d(p=dropout_rate)
        self.drop5 = nn.Dropout3d(p=dropout_rate)
        self.drop7 = nn.Dropout3d(p=dropout_rate)
        self.drop9 = nn.Dropout3d(p=dropout_rate)
        self.drop11 = nn.Dropout3d(p=dropout_rate)
        self.drop13 = nn.Dropout3d(p=dropout_rate)
        
        
    def forward(self, x):
        
        # --------------- Contracting path -------------------
        # 1)
        x = self.conv1a(x) 
        x = F.relu(x)

        x = self.conv1b(x)
        c1b = F.relu(x)
        
        x = self.drop1(c1b)        
        x = F.max_pool3d(x, 2)  #stride=2
        
        # 2)
        x = self.conv2a(x) 
        x = F.relu(x)

        x = self.conv2b(x)
        c2b = F.relu(x)
        
        x = self.drop2(c2b)     
        x = F.max_pool3d(x, 2)  #stride=2
        
        # 3)
        x = self.conv3a(x) 
        x = F.relu(x)

        x = self.conv3b(x)
        c3b = F.relu(x)
        
        x = self.drop3(c3b)
        x = F.max_pool3d(x, 2)  #stride=2
        
        # 4)
        x = self.conv4a(x) 
        x = F.relu(x)

        x = self.conv4b(x)
        c4b = F.relu(x)
        
        x = self.drop4(c4b)   
        x = F.max_pool3d(x, 2)  #stride=2
        
        # 5)
        x = self.conv5a(x) 
        x = F.relu(x)

        x = self.conv5b(x)
        x = F.relu(x)
        
        x = self.drop5(x)
        print(x.shape)
        
        # ----------------- Expansive path --------------------
        # 5)
        x = self.up6a(x)
       
        # 4) 
        x = torch.cat((x, c4b), dim=1)
       
        x = self.conv7a(x)
        x = F.relu(x)
        
        x = self.conv7b(x)
        x = F.relu(x)
        x = self.drop7(x)
        
        x = self.up8a(x)
        
        # 3) 
        x = torch.cat((x,c3b), dim=1)
        
        x = self.conv9a(x)
        x = F.relu(x)
        
        x = self.conv9b(x)
        x = F.relu(x)
        x = self.drop9(x)
        
        x = self.up10a(x)
        
        # 2) 
        x = torch.cat((x,c2b), dim=1)
        
        x = self.conv11a(x)
        x = F.relu(x)
        
        x = self.conv11b(x)
        x = F.relu(x)
        x = self.drop11(x)
        
        x = self.up12a(x)
        
        # 1) 
        x = torch.cat((x,c1b), dim=1)
        
        x = self.conv13a(x)
        x = F.relu(x)
        
        x = self.conv13b(x)
        x = F.relu(x)
        x = self.drop13(x)
        
        x = self.conv_out(x)
        #outputs = F.softmax(x, dim=1)
        
        return x
       