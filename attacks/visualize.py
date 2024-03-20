import torch
import numpy as np
import matplotlib.pyplot as plt

def visualize(x, x_adv, mean, std, x_grad, epsilon, clean_pred, adv_pred, clean_prob, adv_prob):
    x = x.squeeze(0)  # remove batch dimension # B X C H X W ==> C X H X W
    x = x.mul(torch.FloatTensor(std).view(3, 1, 1)).add(
        torch.FloatTensor(mean).view(3, 1, 1)).numpy()  # reverse of normalization op- "unnormalize"
    x = np.transpose(x, (1, 2, 0))  # C X H X W  ==>   H X W X C
    x = np.clip(x, 0, 1)

    x_adv = x_adv.squeeze(0)
    x_adv = x_adv.mul(torch.FloatTensor(std).view(3, 1, 1)).add(
        torch.FloatTensor(mean).view(3, 1, 1)).numpy()  # reverse of normalization op
    x_adv = np.transpose(x_adv, (1, 2, 0))  # C X H X W  ==>   H X W X C
    x_adv = np.clip(x_adv, 0, 1)

    x_grad = x_grad.squeeze(0).numpy()
    x_grad = np.transpose(x_grad, (1, 2, 0))
    x_grad = np.clip(x_grad, 0, 1)

    figure, ax = plt.subplots(1, 3, figsize=(18, 8))
    ax[0].imshow(x)
    ax[0].set_title('Clean Example', fontsize=20)

    ax[1].imshow(x_grad)
    ax[1].set_title('Perturbation', fontsize=20)
    ax[1].set_yticklabels([])
    ax[1].set_xticklabels([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    ax[2].imshow(x_adv)
    ax[2].set_title('Adversarial Example', fontsize=20)

    ax[0].axis('off')
    ax[2].axis('off')

    ax[0].text(1.1, 0.5, "+{}*".format(round(epsilon, 3)), size=15, ha="center",
               transform=ax[0].transAxes)

    ax[0].text(0.5, -0.13, "Prediction: {}\n Probability: {}".format(clean_pred, clean_prob), size=15, ha="center",
               transform=ax[0].transAxes)

    ax[1].text(1.1, 0.5, " = ", size=15, ha="center", transform=ax[1].transAxes)

    ax[2].text(0.5, -0.13, "Prediction: {}\n Probability: {}".format(adv_pred, adv_prob), size=15, ha="center",
               transform=ax[2].transAxes)

    plt.show()