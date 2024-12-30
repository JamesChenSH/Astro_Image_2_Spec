import os, sys, torch, json

from utils.dataset_builder import AstroImageSpecDataset
from model.model_layers import AstroImage2SpecModel
from matplotlib import pyplot as plt

if __name__ == "__main__":
    ds_path = "./datasets/val_ds_10000.pt"
    ds = torch.load(ds_path, weights_only=False)

    # Sample the first image in validation set
    img, spec = ds[0]
    print(img.shape, spec.shape)

    # Print the first star image horizontally
    img_reshaped = torch.reshape(img, (5, 24, 24))
    plt.figure(figsize=(20, 10))
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(img_reshaped[i, :, :])
    plt.savefig("star_img_10k.png")

    spec_gt = spec.numpy()
    # Plot the actual spectrum
    plt.figure()
    plt.plot(spec_gt)
    plt.savefig("spectrum_gt_10k.png")

    # # Generate the spectrum using the model
    # model_path = "./checkpoints/train_2024-12-24_14-48-21/ckpt_ep_999_loss_2_5.pt"        # 200 samples, 1000 epochs, lr=1e-3
    # model_config = "./checkpoints/train_2024-12-24_14-48-21/model_param.json"             # 200 samples, 1000 epochs, lr=1e-3

    model_path = "./checkpoints/train_2024-12-29_20-08-04/ckpt_ep_49_loss_0_0035.pt"
    model_config = "./checkpoints/train_2024-12-29_20-08-04/model_param.json"        

    with open(model_config, "r") as f:
        config = json.load(f)

    model = AstroImage2SpecModel(
        img_depth = 1,                      # [src_len]
        img_size = 2880, 
        spec_depth = 1,                     # [tgt_len]
        spec_len = 3601,
        embedding_dim=config["embedding_dim"],
        
        encoder_head_num=config["encoder_head_num"],
        decoder_head_num=config["decoder_head_num"],

        encoder_ff_dim=config["encoder_ff_dim"],
        decoder_ff_dim=config["decoder_ff_dim"],

        encoder_attn_dropout=config["encoder_attn_dropout"],
        decoder_attn_dropout=config["decoder_attn_dropout"],
        encoder_dropout_rate=config["encoder_dropout_rate"],
        decoder_dropout_rate=config["decoder_dropout_rate"],

        num_enc_layers=config["num_enc_layers"],
        num_dec_layers=config["num_dec_layers"],
        device="cuda"
    ).to('cuda')

    model.load_state_dict(torch.load(model_path, weights_only=True))

    img = img.unsqueeze(0).to('cuda')
    with torch.no_grad():
        generated_spec = model.generate_spectrum(img, process_bar=True)
    
    # Plot the generated spectrum
    plt.figure()
    plt.plot(generated_spec[0].cpu())
    plt.savefig("spectrum_gen_10k.png")

    # Plot Teacher Forced Spectrum
    with torch.no_grad():
        tf_spec = model(img, spec.unsqueeze(0)[:, :-1].to('cuda'))
    plt.figure()
    plt.plot(tf_spec[0].cpu())
    plt.savefig("spectrum_tf_10k.png")
