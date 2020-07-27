from lib import *
from config import *

def make_datapath_list(phase = 'train'):
    rootpath = "./data/hymenoptera_data/"
    target_path = osp.join(rootpath+phase+"/**/*.jpg")
    
    path_list = []
    
    for path in glob.glob(target_path):
        path_list.append(path)
        
    return path_list


# Training
def train_model(net, dataloader_dict, criterior, optimizer, num_epochs):
    # device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
    # nhà nghèo có mỗi con card cũ thì:  PyTorch no longer supports 
    device = torch.device("cpu")
    print("device: ", device)

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs))

        # move network to device
        net.to(device)

        torch.backends.cudnn.benchmark = True
        
        for phase in ["train", "val"]:
            if phase == "train":
                net.train()
            else:
                net.eval()
                
            epoch_loss = 0.0
            epoch_corrects = 0
            
            if (epoch == 0) and (phase == "train"):
                continue
            for inputs, labels in tqdm(dataloader_dict[phase]):
                # move to GPU/CPU
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                #set gradient of optim to be zero
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == "train"):
                    outputs = net(inputs)
                    loss = criterior(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                        
                    epoch_loss += loss.item()*inputs.size(0)
                    epoch_corrects += torch.sum(preds==labels.data)
                    
            
            epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)
            epoch_accuracy = epoch_corrects.double() / len(dataloader_dict[phase].dataset)
                    
            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_accuracy))

    torch.save(net.state_dict(), save_path)

def param_to_update(net):
    param_to_update_1 = []
    param_to_update_2 = []
    param_to_update_3 = []

    update_param_name_1 = ["features"]
    update_param_name_2 = ["classifier.0.weight", "classifier.0.bias", "classifier.3.weight", "classifier.3.bias"]
    update_param_name_3 = ["classifier.6.weight", "classifier.6.bias"]

    for name, param in net.named_parameters():
        if name in update_param_name_1 :
            param.requires_grad = True
            param_to_update_1.append(param)
        elif name in update_param_name_2:
            param.requires_grad = True
            param_to_update_2.append(param)
        elif name in update_param_name_3:
            param.requires_grad = True
            param_to_update_3.append(param)
        else:
            param.requires_grad = False
    return param_to_update_1, param_to_update_2, param_to_update_3

def load_model(net, model_path):
    load_weights = torch.load(model_path)
    net.load_state_dict(load_weights)
    #print(net)

    # load_weights = torch.load(model_path, map_location=("cuda:0", "cpu"))
    # net.load_state_dict(load_weights)

    return net
                