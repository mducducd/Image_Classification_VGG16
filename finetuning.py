from lib import *
from image_transform import ImageTransform
from config import *
from utils import make_datapath_list
from dataset import MyDataset
from utils import make_datapath_list, train_model, param_to_update, load_model
from pred import Predictor
from base_transform import BaseTransform

def main():
    train_list = make_datapath_list(phase="train")
    val_list = make_datapath_list(phase="val")

    # Dataset
    train_dataset = MyDataset(train_list, transform=ImageTransform(resize, mean ,std), phase="train")
    val_dataset = MyDataset(val_list, transform=ImageTransform(resize, mean ,std), phase="val")

    

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False)

    dataloader_dict = {
        "train":train_dataloader, "val":val_dataloader}

    # Network
    use_pretrained = "true"
    net = models.vgg16(pretrained=use_pretrained)
    net.classifier[6] = nn.Linear(in_features=4096, out_features=2)

    # Loss
    criterior = nn.CrossEntropyLoss()

    # Optimizer
    # param_to_update = []

    # update_param_name = ["classifier.6.weight", "classifier.6.bias"]

    # for name, param in net.named_parameters():
    #     if name in update_param_name:
    #         param.requires_grad = True
    #         param_to_update.append(param)
    #         print(name)
    #     else:
    #         param.requires_grad = False
    params1, params2, params3 = param_to_update(net)

    optimizer = optim.SGD([
        {'params': params1, 'lr': 1e-4}, 
        {'params': params2, 'lr': 5e-4},
        {'params': params3, 'lr': 1e-3}, 
    ], momentum=0.9)

    train_model(net, dataloader_dict, criterior, optimizer, num_epochs)

if __name__ == "__main__":
    main()

    # use_pretrained = "true"
    # net = models.vgg16(pretrained=use_pretrained)
    # net.classifier[6] = nn.Linear(in_features=4096, out_features=2)

    # load_model(net , save_path)

    