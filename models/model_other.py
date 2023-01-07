import torchvision.models as models
from models.xception import *
from efficientnet_pytorch import EfficientNet


def choose_model(model_name):

    model = None

    if model_name == 'alexnet':
        model = models.alexnet(num_classes=3)

    elif model_name == 'vgg':
        model = models.vgg19(num_classes=3)

    elif model_name == 'squeezenet':
        model = models.squeezenet1_0(num_classes=3)

    elif model_name == 'densenet':
        model = models.densenet121(num_classes=3)

    elif model_name == 'mobilenet':
        model = models.mobilenet_v2(num_classes=3)

    elif model_name == 'inception':
        model = models.inception_v3(num_classes=3)

    elif model_name == 'googlenet':
        model = models.googlenet(num_classes=3)

    elif model_name == 'xception':
        model = xception(num_classes=3)

    elif model_name == 'resnet':
        model = models.resnet50(num_classes=3)

    elif model_name == 'efficientnet':
        model = EfficientNet.from_name('efficientnet-b0')
        model._fc.out_features = 3


    return model