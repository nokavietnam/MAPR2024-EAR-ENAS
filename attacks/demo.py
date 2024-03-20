import torch.nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
from visualize import visualize
import json

# mean and std will remain same irresptive of the model you use
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def main(eps):
    inception_v3 = models.inception_v3(weights="DEFAULT")  # download and load pretrained inceptionv3 model
    inception_v3.eval()

    img = Image.open("../data/images/cat.jpg")

    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    image_tensor = preprocess(img)  # preprocess an i
    image_tensor = image_tensor.unsqueeze(0)  # add batch dimension.  C X H X W ==> B X C X H X W

    img_variable = Variable(image_tensor, requires_grad=True)  # convert tensor into a variable

    output = inception_v3.forward(img_variable)
    label_idx = torch.max(output.data, 1)[1].numpy()[0]  # get an index(class number) of a largest element

    labels_json = json.load(open("../data/attack/labels.json", "r"))
    labels = {int(idx): label for idx, label in labels_json.items()}
    x_pred = labels[label_idx]

    output_probs = F.softmax(output, dim=1)
    x_pred_prob = round((torch.max(output_probs.data, 1)[0].numpy()[0]) * 100, 4)

    y_true = 282  # tiger cat  ##change this if you change input image
    target = Variable(torch.LongTensor([y_true]), requires_grad=False)

    # perform a backward pass in order to get gradients
    loss = torch.nn.CrossEntropyLoss()
    loss_cal = loss(output, target)
    loss_cal.backward(retain_graph=True)

    x_grad = torch.sign(
        img_variable.grad.data)  # calculate the sign of gradient of the loss func (with respect to input X) (adv)
    x_adversarial = img_variable.data + eps * x_grad  # find adv example using formula shown above
    output_adv = inception_v3.forward(Variable(x_adversarial))  # perform a forward pass on adv example
    x_adv_pred = labels[torch.max(output_adv.data, 1)[1].numpy()[0]]  # classify the adv example
    op_adv_probs = F.softmax(output_adv, dim=1)  # get probability distribution over classes
    adv_pred_prob = round((torch.max(op_adv_probs.data, 1)[0].numpy()[0]) * 100, 4)

    visualize(image_tensor, x_adversarial, mean, std, x_grad, eps, x_pred, x_adv_pred, x_pred_prob, adv_pred_prob, "attack_example.png")


if __name__ == '__main__':
    main(0.001)
