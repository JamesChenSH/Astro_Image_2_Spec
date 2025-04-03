import torch, os, json, transformers
import torch.utils.data
import numpy as np
from argparse import ArgumentParser
from datetime import datetime

from torch.utils.data import Subset

from tqdm import tqdm

from model.model_layers import AstroImage2SpecModel
from utils.dataset_builder import AstroImageSpecDataset

torch.manual_seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

class  AstroImage2Spec():
    '''
    The Wrapper class for Audio2Image model. We can define the structure of model here
    
    model input: [audio_timeline, audio_fourier] 
    model output: [img_pixel, 0-255]
    '''
    def __init__(self,
        img_depth:int = 1,                      # [src_len]
        img_len:int = 2880, 
        spec_depth:int = 1,                     # [tgt_len]
        spec_len:int = 3601,                    # 3600 from spectrum, 1 for the <sos> token
        device:str = 'cuda',                    # 'cuda' or 'cpu' or 'mps'
        embedding_dim:int = 256,                # 1024 for optimal
        encoder_head_num:int = 8,               
        decoder_head_num:int = 8,
        encoder_ff_dim:int = 4*256,             # 4*1024 for optimal
        decoder_ff_dim:int = 4*256,             # 4*1024 for optimal
        encoder_dropout_rate:float = 0.1, 
        decoder_dropout_rate:float = 0.1,
        encoder_attn_dropout:float = 0.0,
        decoder_attn_dropout:float = 0.0, 
        num_enc_layers:int = 3,                 # 12 for optimal
        num_dec_layers:int = 3,                 # 12 for optimal  

        epochs:int = 10,
        patience:int = 5,
        lr: float = 1e-3,
        warmup: int = 11250
    ):
        """
        This is the main model for the Audio 2 Image project. We only need to build this once
        in the training script. The hyperparameters are set to default similar to that of GPT-2.
        
        With the defualt settings, there are 176,893,184 parameters in the model with FP32 precision, 
        requiring around 10-12 GB of memory on a GPU with batch size 32.
        
        img_depth: int
            Depth of the image data, currently 1 since flatten
        img_len: int
            Length of the image data, currently 2880 
        spec_depth: int
            Depth of the spectrum data, currently 1 since flatten
        spec_len: int
            Length of the spectrum data, currently 3600
        device: str
            Device to run the model on, either 'cuda' or 'cpu' or 'mps'
        embedding_dim: int
            Dimension of the embedding
        encoder_head_num: int
            Number of Multi-Head-Attention heads in the encoder
        decoder_head_num: int
            Number of MH-Attention heads in the decoder
        encoder_ff_dim: int
            Dimension of the feed forward layer in the encoder
        decoder_ff_dim: int
            Dimension of the ff layer in the decoder
        encoder_dropout_rate: float
            FF layer dropout rate in the encoder
        decoder_dropout_rate: float
            FF layer dropout rate in the decoder
        encoder_attn_dropout: float
            Attention dropout rate in the encoder
        decoder_attn_dropout: float
            Attention dropout rate in the decoder
        num_enc_layers: int
            Number of encoder layers
        num_dec_layers: int
            Number of decoder layers
        """
        
        # Model Parameters
        self.img_depth = img_depth
        self.img_len = img_len
        self.spec_dpeth = spec_depth
        self.spec_len = spec_len
        self.embedding_dim = embedding_dim
        self.encoder_head_num = encoder_head_num
        self.decoder_head_num = decoder_head_num
        self.encoder_ff_dim = encoder_ff_dim
        self.decoder_ff_dim = decoder_ff_dim
        self.encoder_dropout_rate = encoder_dropout_rate
        self.decoder_dropout_rate = decoder_dropout_rate
        self.encoder_attn_dropout = encoder_attn_dropout
        self.decoder_attn_dropout = decoder_attn_dropout
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers
        
        if device == 'cuda' and torch.cuda.is_available():
            self.device = "cuda"
        elif device == 'mps':
            self.device = "mps"
        else:
            print("CUDA Device not available, using CPU")
            self.device = 'cpu'
        
        self.model = AstroImage2SpecModel(
            self.img_depth,
            self.img_len,
            self.spec_dpeth,
            self.spec_len,  
            self.embedding_dim, 
            self.encoder_head_num, 
            self.decoder_head_num, 
            self.encoder_ff_dim, 
            self.decoder_ff_dim, 
            self.encoder_dropout_rate, 
            self.decoder_dropout_rate, 
            self.encoder_attn_dropout, 
            self.decoder_attn_dropout, 
            self.num_enc_layers, 
            self.num_dec_layers,
            self.device
        ).to(self.device)
        
        print(f"Model created on device: {self.device}")
        
        self.encoder = self.model.encoder
        self.decoder = self.model.decoder
        
        # HyperParameters
        self.label_smoothing = 0.1
        self.learning_rate = lr
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.95), eps=1e-9, weight_decay=0.1)
        # self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: self.lr_scheduler(self.embedding_dim, step, warmup=11250))
        self.scheduler = transformers.get_scheduler("cosine", self.optimizer, num_warmup_steps=warmup, num_training_steps=warmup*epochs)
        self.criterion = torch.nn.MSELoss(reduction='mean')     
        self.epochs = epochs
        self.patience = patience
        
        # Init Metadata for Saving
        self.training_start_time = None

        
    def lr_scheduler(self, dim_model: int, step:int, warmup:int):
        if step == 0:
            step = 1
        return (dim_model ** -0.5) * min(step ** -0.5, step * warmup ** -1.5)

   
    def train(
        self,
        training_dataloader:torch.utils.data.DataLoader,
        val_dataloader:torch.utils.data.DataLoader,
    ) -> None:
        '''
        Parameters:
        input_audio: np.ndarray
            Input audio data, 3D array of shape (num_samples, audio_size, audio_val)
        output_imgs: np.ndarray
            Output image data, 3D array of shape (num_samples, img_size, img_val)
        '''
        
        # Start time
        self.training_start_time = datetime.now()
        
        self.model.to(self.device)
        self.criterion.to(self.device)
        
        cached_param = None
        lowest_val_loss = float('inf')
        wait_count = 0
        
        for epoch in range(self.epochs):
            self.model.train()
            
            total_loss = 0
            print(f"== Epoch: {epoch}, Device: {self.device} ==")

            pbar = tqdm(training_dataloader)

            for img, spectrum in pbar:

                img = img.to(self.device)
                spectrum = spectrum.to(self.device)
                
                # Input a shifted out_image to model as well as input audio
                output = self.model(img, spectrum[:, :-1]).squeeze(-1)
                # Outputs a predicted image
                loss = self.criterion(output, spectrum[:, 1:])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                total_loss += loss.item()

                pbar.set_description(f"Cur Loss: {round(loss.item(), 4)}")
            
            print(f"== Training Loss: {total_loss / len(train_dataloader)}, Device: {self.device}")
            

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for i, (img, spectrum) in enumerate(val_dataloader):
                    # Compare the predicted image with the actual image with some function
                    
                    img = img.to(self.device)
                    spectrum = spectrum.to(self.device)
                    
                    gen_spec = self.model(img, spectrum[:, :-1])
                    # Use average MSE over the epoch
                    loss = self.criterion(gen_spec.reshape(-1, gen_spec.shape[-1]).squeeze(), spectrum[:, 1:].contiguous().view(-1))
                    val_loss += loss.item()
                
                val_loss /= len(val_dataloader)
                
                if val_loss < lowest_val_loss:
                    lowest_val_loss = val_loss
                    wait_count = 0
                    cached_param = self.model.state_dict()
                else:
                    wait_count += 1
                    print(f"Waiting: {wait_count}")
                    if wait_count == self.patience:
                        print("Checkpoint Saved")
                        self.save(epoch, lowest_val_loss, base_path='./checkpoints', ckpt=cached_param)
            
            print(f"== Validation Loss: {val_loss}, Device: {self.device}")
        self.save(epoch, lowest_val_loss, base_path='./checkpoints', ckpt=cached_param)
        
        print(f"Training Complete")
        return total_loss / len(training_dataloader), val_loss
            

    # def test(
    #     self,
    #     testing_dataloader:torch.utils.data.DataLoader
    # ):
    #     self.model.to(self.device)
    #     self.criterion.to(self.device)
        
    #     self.model.eval()
        
    #     test_loss = 0
        
    #     with torch.no_grad():
    #         for img, spectrum in tqdm(testing_dataloader):
                
    #             spectrum = spectrum.to(self.device)
    #             img = img.to(self.device)
            
    #             gen_img = self.model.generate_spectrum(img)

    #             gen_img_np = gen_img.detach().cpu().numpy().astype(np.float32)
    #             img_np = img.detach().cpu().numpy().astype(np.float32) 

    #             loss = self.validation_criterion(gen_img_np, img_np, data_range=259.0)
    #             test_loss += loss/test_dataloader.batch_size
        
    #     print(f"Test Loss: {test_loss}, Device: {self.device}")             
    
    
    def save(self, cur_epoch: int=0, cur_loss: float=0.0, base_path: str='./checkpoints', ckpt = None) -> None:
        '''
        Save the model to a file
        '''
        
        if not os.path.isdir(base_path):
            os.mkdir(base_path)
        
        # Compose model directory name
        model_dir = f"{base_path}/train_{self.training_start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
        
        # If never saved before, save model parameters in json
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
            # Compose model parameter json
            model_param = {
                'embedding_dim': self.embedding_dim,
                'encoder_head_num': self.encoder_head_num,
                'decoder_head_num': self.decoder_head_num,
                'encoder_ff_dim': self.encoder_ff_dim,
                'decoder_ff_dim': self.decoder_ff_dim,
                'encoder_dropout_rate': self.encoder_dropout_rate,
                'decoder_dropout_rate': self.decoder_dropout_rate,
                'encoder_attn_dropout': self.encoder_attn_dropout,
                'decoder_attn_dropout': self.decoder_attn_dropout,
                'num_enc_layers': self.num_enc_layers,
                'num_dec_layers': self.num_dec_layers,
                'lr': self.learning_rate,
                'epochs': self.epochs,
                'patience': self.patience
            }
            # Save model parameters
            with open(f"{model_dir}/model_param.json", 'w') as f:
                json.dump(model_param, f)
        
        # Save model
        
        # Compose model name
        if ckpt is not None:
            model_name = f"{model_dir}/ckpt_ep_{cur_epoch}_loss_{str(round(cur_loss, 4)).replace('.', '_')}.pt" 
            torch.save(self.model.state_dict(), model_name)
        else:
            model_name = f"{model_dir}/model_ep_{cur_epoch}_loss_{str(round(cur_loss, 4)).replace('.', '_')}.pt"
            torch.save(self.model.state_dict(), model_name)
            


if __name__ == "__main__":

    parser = ArgumentParser()
    
    parser.add_argument("--use_model", type=str, default=None, help="Path to the model's folder")
    parser.add_argument("--use_base_dataset", type=str, default='./datasets/AstroImg2Spec_ds_1000.pt', help="Path to the base dataset") 
    parser.add_argument("--use_train_dataset", type=str, default=None, help="Customize the training dataset")
    parser.add_argument("--use_val_dataset", type=str, default=None, help="Customize the validation dataset")
    

    config = {
        'batch size': 4,
        'train ratio': 0.9,
        'validation ratio': 0.1,
        'device': 'cuda',

        'embedding_dim': 512,
        'encoder_head_num': 8,
        'decoder_head_num': 8,

        'encoder_ff_dim': 4*512,
        'decoder_ff_dim': 4*512,
        
        'num_enc_layers': 6,
        'num_dec_layers': 6,

        'lr': 1e-5,
        'epochs': 50,
        'patience': 5,
        'warmup': 11250
    }

    
    a2i_core = AstroImage2Spec(
        embedding_dim=config['embedding_dim'],
        encoder_head_num=config['encoder_head_num'],
        decoder_head_num=config['decoder_head_num'],
        encoder_ff_dim=config['encoder_ff_dim'],
        decoder_ff_dim=config['decoder_ff_dim'],
        lr=config['lr'],
        epochs=config['epochs'],
        device=config['device'],
        num_enc_layers=config['num_enc_layers'],
        num_dec_layers=config['num_dec_layers'],
        patience=config['patience'],
        warmup=config['warmup']
    )

    # Load the dataset
    N = 10000
    ds_path = f"datasets/AstroImg2Spec_ds_{N}.pt"
    ds = torch.load(ds_path, weights_only=False)

    # Split Train, Val, Test
    train_size = int(config['train ratio']*len(ds))
    # train_size = 4000
    val_size = int(config['validation ratio']*len(ds))
    test_size = len(ds) - train_size - val_size

    train, val = torch.utils.data.random_split(ds, [train_size, val_size])
    torch.save(train, f"datasets/train_ds_{N}.pt")
    torch.save(val, f"datasets/val_ds_{N}.pt")
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=config['batch size'], shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=config['batch size'], shuffle=True)   
    
    
    # Chack size of model
    total_params = sum(p.numel() for p in a2i_core.model.parameters())
    print(f"Number of parameters: {total_params}")
    
    # Train
    train_loss, val_loss = a2i_core.train(train_dataloader, val_dataloader)
    
    # Save the model
    a2i_core.save(cur_epoch=config['epochs'], cur_loss=val_loss, ckpt=None)
    
    # # Test
    # a2i_core.test(test_dataloader)
