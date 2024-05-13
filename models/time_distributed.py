import torch
import torch.nn as nn

import timm

#using code from issue https://discuss.pytorch.org/t/timedistributed-cnn/51707/12

class TimeDistributed(nn.Module):

    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        ''' x size: (batch_size, time_steps, in_channels, height, width) '''
        B, T, F, H, W = x.size()
        c_in = x.view(B * T, F, H, W)
        c_out = self.module(c_in)
        llayer = self.module[-1] if isinstance(self.module, nn.Sequential) else self.module
        if isinstance(llayer, nn.Flatten):
           r_in = c_out.view(B*T, H*W, F)
           return r_in
        r_in = c_out.view(B, T, c_out.shape[1], c_out.shape[2], c_out.shape[3])
        return r_in


extractor = TimeDistributed(timm.create_model('resnet18', pretrained=True, num_classes=0, global_pool=''))


layers= [TimeDistributed(nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU())),
         TimeDistributed(nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64),nn.ReLU(),
                       nn.MaxPool2d(kernel_size = 2, stride = 2)))]

layers2 = [TimeDistributed(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)),
           TimeDistributed(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))]

class VGG16Extractor(nn.Module):

    N_LAYERS =13

    def __init__(self):
        super(VGG16Extractor, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())

        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())

        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())

        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())

        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())

        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())

        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())

        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        return out

    @classmethod
    def get_sub_extractor(cls, module: nn.Module, l_layer: int):

        if not isinstance(module, VGG16Extractor):
            raise ValueError('module is not the VGG16Extractor')

        if l_layer < 1 or l_layer > VGG16Extractor.N_LAYERS:
            raise ValueError(f'l_layer must be between from 1 to {VGG16Extractor.N_LAYERS}')

        class SubVGG16Extractor(nn.Module):
            def __init__(self, module,  l_layer ):
                super(SubVGG16Extractor, self).__init__()
                self.layers = list(module.children())[:l_layer]

            def forward(self, x):
                out = x
                for layer in self.layers:
                    out = layer(out)
                return out
        subVGG16  = SubVGG16Extractor(module, l_layer)
        return subVGG16

class TimeDistributedCollisionModel(nn.Module):

    def __init__(self, extractor = 'resnet18', embeded_sz=512, n_classes=1, is_trainable=True, lstm_units=32):
        super(TimeDistributedCollisionModel, self).__init__()
        if extractor.lower() == 'vgg16':
           vgg16Extractor = VGG16Extractor()
           self.extractor = TimeDistributed(VGG16Extractor.get_sub_extractor(vgg16Extractor, 8))
           self.embeded_sz = 512
           self.extractor_train = True
        else:
           self.extractor = TimeDistributed(timm.create_model(extractor,
                                                   pretrained=True, num_classes=0, global_pool=''))
           self.embeded_sz = embeded_sz

        self.extr_trainable = is_trainable
        self.n_classes = n_classes
        self.lstm_units = lstm_units

        self.global_pooling = TimeDistributed(nn.AdaptiveAvgPool2d(1))

        self.chanel_pooling = nn.Conv1d(10, out_channels=1, kernel_size=3, padding=1)

        self.lstm1 = nn.LSTM(self.embeded_sz, self.lstm_units,
                                                bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(2*self.lstm_units, self.lstm_units,
                                                bidirectional=True, batch_first=True)
        self.out_layer = nn.Linear(2*self.lstm_units, n_classes)

    def forward_extractor(self, x):
        out = self.extractor(x)
        out = self.global_pooling(out)
        B, T, F, _, _ = out.size()
        out = out.reshape(B, T, F)
        return out

    def forward(self, x):
        out = x
        if self.extr_trainable:
            with torch.no_grad():
                out = self.forward_extractor(out)
        else:
           out = self.forward_extractor(out)

        out, _ = self.lstm1(out)
        out = self.chanel_pooling(out)
        out = self.out_layer(out)
        out = torch.squeeze(out)
        return out






