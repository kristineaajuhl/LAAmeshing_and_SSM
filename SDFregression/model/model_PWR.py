"""
PWR model using convolutions and upsampling 
FROM PWR MODEL 08-06-2020
15-06-2020
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class my3DUnet(BaseModel):
    def __init__(self, n_features = 64, n_classes = 1, image_size = 64, dropout_rate=0.05):
        super().__init__()
        self.features = n_features
        self.in_channels = 1
        self.n_classes = n_classes             
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
        self.up6a2 = nn.Conv3d(self.features * 16, self.features * 8, kernel_size=3, stride=1, padding=1)
        self.conv7a = nn.Conv3d(self.features * 16, self.features * 8, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.conv7b = nn.Conv3d(self.features * 8, self.features * 8, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.up8a = nn.ConvTranspose3d(self.features * 8, self.features * 4, kernel_size=2, stride=2, padding=0)
        self.up8a2 = nn.Conv3d(self.features * 8, self.features * 4, kernel_size=3, stride=1, padding=1)
        self.conv9a = nn.Conv3d(self.features * 8, self.features * 4, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.conv9b = nn.Conv3d(self.features * 4, self.features * 4, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.up10a = nn.ConvTranspose3d(self.features * 4, self.features * 2, kernel_size=2, stride=2, padding=0)
        self.up10a2 = nn.Conv3d(self.features * 4, self.features * 2, kernel_size=3, stride=1, padding=1)
        self.conv11a = nn.Conv3d(self.features * 4, self.features * 2, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.conv11b = nn.Conv3d(self.features * 2, self.features * 2, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.up12a = nn.ConvTranspose3d(self.features * 2, self.features, kernel_size=2, stride=2, padding=0)
        self.up12a2 = nn.Conv3d(self.features * 2, self.features, kernel_size=3, stride=1, padding=1)
        self.conv13a = nn.Conv3d(self.features * 2, self.features, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.conv13b = nn.Conv3d(self.features, self.features, kernel_size=3, stride=1, padding=1, padding_mode='zeros')

        #self.conv_out = nn.Conv3d(self.features, self.n_classes, kernel_size=1, stride=1, padding=0)
        self.conv_out = nn.Conv3d(self.features, 1, kernel_size=1, stride=1, padding=0)

        self.drop1 = nn.Dropout3d(p=dropout_rate)
        self.drop2 = nn.Dropout3d(p=dropout_rate)
        self.drop3 = nn.Dropout3d(p=dropout_rate)
        self.drop4 = nn.Dropout3d(p=dropout_rate)
        self.drop5 = nn.Dropout3d(p=dropout_rate)
        self.drop7 = nn.Dropout3d(p=dropout_rate)
        self.drop9 = nn.Dropout3d(p=dropout_rate)
        self.drop11 = nn.Dropout3d(p=dropout_rate)
        self.drop13 = nn.Dropout3d(p=dropout_rate)

        self.bn1a = nn.BatchNorm3d(self.features)
        self.bn1b = nn.BatchNorm3d(self.features)
        self.bn2a = nn.BatchNorm3d(self.features * 2)
        self.bn2b = nn.BatchNorm3d(self.features * 2)
        self.bn3a = nn.BatchNorm3d(self.features * 4)
        self.bn3b = nn.BatchNorm3d(self.features * 4)
        self.bn4a = nn.BatchNorm3d(self.features * 8)
        self.bn4b = nn.BatchNorm3d(self.features * 8)
        self.bn5a = nn.BatchNorm3d(self.features * 16)
        self.bn5b = nn.BatchNorm3d(self.features * 16)

        self.bn7a = nn.BatchNorm3d(self.features * 8)
        self.bn7b = nn.BatchNorm3d(self.features * 8)

        self.bn9a = nn.BatchNorm3d(self.features * 4)
        self.bn9b = nn.BatchNorm3d(self.features * 4)

        self.bn11a = nn.BatchNorm3d(self.features * 2)
        self.bn11b = nn.BatchNorm3d(self.features * 2)

        self.bn13a = nn.BatchNorm3d(self.features)
        self.bn13b = nn.BatchNorm3d(self.features)

        # self.bn6a = nn.BatchNorm3d(self.features)
        # self.bn8a = nn.BatchNorm3d(self.features)
        # self.bn10a = nn.BatchNorm3d(self.features)
        # self.bn12a = nn.BatchNorm3d(self.features)

    def forward(self, x):
        # assuming input images are 128 x 128 x 128 x 1
        # --------------- Contracting path -------------------

        # Kristines: Unet_GPU_SD3_005
        # conv1a = Conv3D(64, kernel_size, activation='relu', padding='same')(inputs)
        x = self.conv1a(x)  # x: (128 x 128 x 128 x 64)
        x = F.relu(x)
        x = self.bn1a(x)

        # conv1b = Conv3D(64, kernel_size, activation='relu', padding='same')(conv1a)
        x = self.conv1b(x)  # x: (128 x 128 x 128 x 64)
        # c1b = F.relu(x)
        x = F.relu(x)
        c1b = self.bn1b(x)

        # drop1 = SpatialDropout3D(0.05)(conv1b)
        x = self.drop1(c1b)  # x: (128 x 128 x 128 x 64)

        # pool1 = MaxPooling3D(pool_size=(2, 2, 2))(drop1)
        x = F.max_pool3d(x, 2)  # x: (64 x 64 x 64 x 64)

        # conv2a = Conv3D(128, kernel_size, activation='relu', padding='same')(pool1)
        x = self.conv2a(x)  # x: (64 x 64 x 64 x 128)
        x = F.relu(x)
        x = self.bn2a(x)

        # conv2b = Conv3D(128, kernel_size, activation='relu', padding='same')(conv2a)
        x = self.conv2b(x)  # x: (64 x 64 x 64 x 128)
        # c2b = F.relu(x)
        x = F.relu(x)
        c2b = self.bn2b(x)

        # drop2 = SpatialDropout3D(0.05)(conv2b)
        x = self.drop2(c2b)  # x: (64 x 64 x 64 x 128)

        # pool2 = MaxPooling3D(pool_size=(2, 2, 2))(drop2)
        x = F.max_pool3d(x, 2)  # x: (32 x 32 x 32 x 128)

        # conv3a = Conv3D(256, kernel_size, activation='relu', padding='same')(pool2)
        x = self.conv3a(x)  # x: (32 x 32 x 32 x 256)
        x = F.relu(x)
        x = self.bn3a(x)

        # conv3b = Conv3D(256, kernel_size, activation='relu', padding='same')(conv3a)
        x = self.conv3b(x)  # x: (32 x 32 x 32 x 256)
        # c3b = F.relu(x)
        x = F.relu(x)
        c3b = self.bn3b(x)

        # drop3 = SpatialDropout3D(0.05)(conv3b)
        x = self.drop3(c3b)

        # pool3 = MaxPooling3D(pool_size=(2, 2, 2))(drop3)
        x = F.max_pool3d(x, 2)  # x: (16 x 16 x 16 x 256)

        # conv4a = Conv3D(512, kernel_size, activation='relu', padding='same')(pool3)
        x = self.conv4a(x)  # x: (16 x 16 x 16 x 512)
        x = F.relu(x)
        x = self.bn4a(x)

        # conv4b = Conv3D(512, kernel_size, activation='relu', padding='same')(conv4a)
        x = self.conv4b(x)  # x: (16 x 16 x 16 x 512)
        # c4b = F.relu(x)
        x = F.relu(x)
        c4b = self.bn4b(x)

        # drop4 = SpatialDropout3D(0.05)(conv4b)
        x = self.drop4(c4b)

        # pool4 = MaxPooling3D(pool_size=(2, 2, 2))(drop4)
        x = F.max_pool3d(x, 2)  # x: (8 x 8 x 8 x 512)

        # conv5a = Conv3D(1024, kernel_size, activation='relu', padding='same')(pool4)
        x = self.conv5a(x)  # x: (8 x 8 x 8 x 1024)
        x = F.relu(x)
        x = self.bn5a(x)

        # conv5b = Conv3D(1024, kernel_size, activation='relu', padding='same')(conv5a)
        x = self.conv5b(x)  # x: (8 x 8 x 8 x 1024)
        x = F.relu(x)
        x = self.bn5b(x)

        # drop5 = SpatialDropout3D(0.05)(conv5b)
        x = self.drop5(x)  # x: (8 x 8 x 8 x 1024)

        # ----------------- Expansive path --------------------
        # kernel_size_trans = (2, 2, 2)
        # strides_trans = (2, 2, 2)
        # up6a = Conv3DTranspose(512, kernel_size_trans, strides=strides_trans, padding='same')(drop5)
        # x = self.up6a(x)  # x: (16 x 16 x 16 x 512)
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
        x = self.up6a2(x)

        # up6b = concatenate([up6a, conv4b], axis=4)
        x = torch.cat((x, c4b), dim=1)

        # conv7a = Conv3D(512, kernel_size, activation='relu', padding='same')(up6b)
        x = self.conv7a(x)  # x: (16 x 16 x 16 x 512)
        x = F.relu(x)
        x = self.bn7a(x)

        # conv7b = Conv3D(512, kernel_size, activation='relu', padding='same')(conv7a)
        x = self.conv7b(x)  # x: (16 x 16 x 16 x 512)
        x = F.relu(x)
        x = self.bn7b(x)

        # drop7 = SpatialDropout3D(0.05)(conv7b)
        x = self.drop7(x)

        # up8a = Conv3DTranspose(256, kernel_size_trans, strides=strides_trans, padding='same')(drop7)
        # x = self.up8a(x)  # x: (32 x 32 x 32 x 256)
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
        x = self.up8a2(x)

        # up8b = concatenate([up8a, conv3b], axis=4)
        x = torch.cat((x, c3b), dim=1)

        # conv9a = Conv3D(256, kernel_size, activation='relu', padding='same')(up8b)
        x = self.conv9a(x)  # x: (32 x 32 x 32 x 256)
        x = F.relu(x)
        x = self.bn9a(x)

        # conv9b = Conv3D(256, kernel_size, activation='relu', padding='same')(conv9a)
        x = self.conv9b(x)  # x: (32 x 32 x 32 x 256)
        x = F.relu(x)
        x = self.bn9b(x)

        # drop9 = SpatialDropout3D(0.05)(conv9b)
        x = self.drop9(x)

        # up10a = Conv3DTranspose(128, kernel_size_trans, strides=strides_trans, padding='same')(drop9)
        # x = self.up10a(x)  # x: (64 x 64 x 64 x 128)
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
        x = self.up10a2(x)

        # up10b = concatenate([up10a, conv2b], axis=4)
        x = torch.cat((x, c2b), dim=1)

        # conv11a = Conv3D(128, kernel_size, activation='relu', padding='same')(up10b)
        x = self.conv11a(x)  # x: (64 x 64 x 64 x 128)
        x = F.relu(x)
        x = self.bn11a(x)

        # conv11b = Conv3D(128, kernel_size, activation='relu', padding='same')(conv11a)
        x = self.conv11b(x)  # x: (64 x 64 x 64 x 128)
        x = F.relu(x)
        x = self.bn11b(x)

        # drop11 = SpatialDropout3D(0.05)(conv11b)
        x = self.drop11(x)

        # up12a = Conv3DTranspose(64, kernel_size_trans, strides=strides_trans, padding='same')(drop11)
        # x = self.up12a(x)  # x: (128 x 128 x 128 x 64)
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
        x = self.up12a2(x)

        # up12b = concatenate([up12a, conv1b], axis=4)
        x = torch.cat((x, c1b), dim=1)

        # conv13a = Conv3D(64, kernel_size, activation='relu', padding='same')(up12b)
        x = self.conv13a(x)  # x: (128 x 128 x 128 x 64)
        x = F.relu(x)
        x = self.bn13a(x)

        # conv13b = Conv3D(64, kernel_size, activation='relu', padding='same')(conv13a)
        x = self.conv13b(x)  # x: (128 x 128 x 128 x 64)
        # x = F.relu(x)
        # x = self.bn13b(x)

        # drop13 = SpatialDropout3D(0.05)(conv13b)
        x = self.drop13(x)

        # out = Conv3D(1, (1, 1, 1), activation='linear', padding='same')(drop13)
        # outputs: (128 x 128 x 128 x 1)
        outputs = self.conv_out(x)

        return outputs
