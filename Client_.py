from mobilnet import mobilenet_v2, tiny_mobilenet_v2
from torch import nn
import torch
import random
import numpy as np
from torchvision import datasets, transforms
import pickle
import logging
import colorlog
from typing import List

def init_logger(dunder_name) -> logging.Logger:
    log_format = (
        '[%(asctime)19s - '
        '%(name)s - '
        '%(levelname)s]  '
        '%(message)s'
    )
    bold_seq = '\033[1m'
    colorlog_format = (
        f'{bold_seq} '
        '%(log_color)s '
        f'{log_format}'
    )
    colorlog.basicConfig(format=colorlog_format)
    logger = logging.getLogger(dunder_name)

    logger.setLevel(logging.INFO)

    # Output full log
    fh = logging.FileHandler('app.log')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(log_format)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger



def con_freeze(model: torch.nn.Module, names: List[str] = ["classifier.1"], freeze=False):
    for n, param in model.named_modules():
        if n in names:
            continue
        param.requires_grad = freeze

def surgery(model:torch.nn.Module, num_unseen:int, name:str = "classifier.1"):
    for n, module in model.named_modules():
        if n == name:
            m = module

    if m.weight != None:
        #TODO device to config
        unseen_weights = torch.empty(num_unseen, m.weight.shape[1], device="cuda:0")
        torch.nn.init.xavier_normal_(unseen_weights)
        tmp = torch.cat([m.weight, unseen_weights], dim=0)
        m.weight.data = tmp

    if m.bias != None:
        unseen_weights = torch.randn(num_unseen, device="cuda:0")
        #torch.nn.init.xavier_normal(unseen_weights)
        tmp = torch.cat([m.bias, unseen_weights], dim=0)
        m.bias.data = tmp



def inference(model, test_dataset, device):
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=100,
                                             shuffle=False)
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            # Inference
            outputs = model(images)
            # print(outputs)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            class_correct = torch.sum(torch.eq(pred_labels, labels)).item()
            correct += class_correct
            total += len(labels)
    # for i in range(10): class_acc[i]/=1000
    accuracy = correct / len(test_dataset)
    del images
    del labels
    # print(class_acc)
    return accuracy, loss / len(iter(testloader))


def get_fewshot_dataset(root: str, trasnform: transforms, class_to_idx, shot=10, base=True):
    dataset = datasets.ImageFolder(root, transform=trasnform)
    data, labels = dataset.imgs, dataset.targets
    indices_class = [[] for _ in range(len(dataset.classes))]

    for i, lab in enumerate(labels):
        indices_class[lab].append(i)

    origin_labels = []
    if base:
        for key in dataset.class_to_idx:
            origin_labels.append(class_to_idx[key])

        for idx, label in enumerate(labels):
            labels[idx] = origin_labels[label]
    else:
        for idx, label in enumerate(labels):
            labels[idx] += 1000

    """10-shot dataset 만들기"""
    def get_images(c, n):  # get random n images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        # idx_shuffle = indices_class[c][:n]
        return idx_shuffle

    all_class_idx = list()

    for c in range(len(dataset.classes)):
        all_class_idx.append(get_images(c, shot))  # 10 shot
    all_class_idx = np.concatenate(all_class_idx)

    data_list = list()
    for class_idx in all_class_idx:
        data_list.append((data[class_idx][0], labels[class_idx]))

    dataset.imgs = data_list
    dataset.samples = data_list
    dataset.targets = list(np.array(labels)[all_class_idx])
    return dataset


def seed_fix(logger, seed=2):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"seed: {seed}")


if __name__ == '__main__':

    pruning_ratio = 0.05
    device = 'cuda:0'
    logger = init_logger(__name__)
    seed_fix(logger)

    logger.info(f"pruning ratio: {pruning_ratio}")

    #model
    model = mobilenet_v2(pretrained=False, pruning_ratio=1 - pruning_ratio, width_mult=0.75)
    m = torch.load("p5.pt")
    model.load_state_dict(m)
    model.to(device)


    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose([transforms.RandomCrop(224, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean, std),
                                          ])

    valid_transform = transforms.Compose([transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean, std),
                                          ])

    """Imagenet """
    # key = class name, value = index
    with open('class_to_idx.pickle', 'rb') as f:
        imagenet_class_to_idx = pickle.load(f)

    # 10 classes corresponding to a single clint
    base_dataset = get_fewshot_dataset("/home/khj/data/deepswitch/jetson1_imagenet_10/train", train_transform, imagenet_class_to_idx)
    test_dataset = get_fewshot_dataset("/home/khj/data/deepswitch/jetson1_imagenet_10/validation", valid_transform, imagenet_class_to_idx)
    logger.info(f"base_class: {base_dataset.classes}")

    """Novel Class """
    novel_dataset = get_fewshot_dataset("/home/khj/data/deepswitch/Novel_dataset/train", train_transform, imagenet_class_to_idx, base=False)
    novel_test_dataset = get_fewshot_dataset("/home/khj/data/deepswitch/Novel_dataset/validation", valid_transform, imagenet_class_to_idx, base=False)
    logger.info(f"num novel classes: {len(novel_dataset.classes)}")
    logger.info(f"novel train dataset size:  {len(novel_dataset.imgs)}")
    # novel_class_idx 0->1000, 1->1001
    concated_dataset = torch.utils.data.ConcatDataset([base_dataset, novel_dataset])
    train_loader = torch.utils.data.DataLoader(concated_dataset,
                                               batch_size=32, shuffle=True, num_workers=4, pin_memory=True)#, collate_fn=lambda x:x)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=4e-5)
    criterion = nn.CrossEntropyLoss()

    epochs = 100

    print()
    logger.info(f"training begin...")

    surgery(model, len(novel_dataset.classes))
    logger.info(f"surgery done")

    con_freeze(model)
    logger.info(f"freeze done")

    for epoch in range(epochs):
        model.train()
        g_loss = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images.requires_grad = True
            labels.requires_grad = False
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)
            g_loss = loss.item()

            loss.backward()

            optimizer.step()

            torch.cuda.empty_cache()

        logger.info(f"epoch {epoch}, loss {g_loss:.4f}")
        if epoch % 5 == 0:
            accuracy, loss = inference(model, test_dataset, device=device)
            print('Test Base Accuracy: {:.2f}%'.format(100 * accuracy))
            print(f'Test Base Loss: {loss} \n')

            accuracy, loss = inference(model, novel_test_dataset, device=device)
            print('Test Novel Accuracy: {:.2f}%'.format(100 * accuracy))
            print(f'Test Novel Loss: {loss} \n')