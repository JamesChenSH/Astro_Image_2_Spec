import PIL, torch
from matplotlib import pyplot as plt

from model.model_layers import AstroImage2SpecModel


def img_2_spec(image:torch.Tensor, model:AstroImage2SpecModel, gt_spec=None):
    
    # Predict the spectrum
    gen_spectrum = model.generate_spectrum(image.unsqueeze(-1))
    plt.plot(gen_spectrum)
    if gt_spec is not None:
        plt.plot(gt_spec)
    plt.show()
    
    return gen_spectrum


if __name__ == "__main__":
    ds = torch.load('./datasets/AstroImg2Spec_ds_1000.pt')
    model = torch.load('./model.pt')
    
    img_2_spec(ds.imgs[0], model, ds.specs[0])