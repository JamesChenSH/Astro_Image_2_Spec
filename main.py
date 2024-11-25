import torch
import torch.utils.data
from torch.utils.data import Subset

from tqdm import tqdm

from model.model_layers import AstroImage2SpecModel
from utils.dataset_builder import AstroImageSpecDataset

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
        spec_len:int = 3600,
        device:str = 'cuda',                    # 'cuda' or 'cpu' or 'mps'
        embedding_dim:int = 512,                # 1024 for optimal
        encoder_head_num:int = 2,               
        decoder_head_num:int = 2,
        encoder_ff_dim:int = 4*512,             # 4*1024 for optimal
        decoder_ff_dim:int = 4*512,             # 4*1024 for optimal
        encoder_dropout_rate:float = 0.1, 
        decoder_dropout_rate:float = 0.1,
        encoder_attn_dropout:float = 0.0,
        decoder_attn_dropout:float = 0.0, 
        num_enc_layers:int = 2,                 # 12 for optimal
        num_dec_layers:int = 2                  # 12 for optimal  
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
        self.learning_rate = 1
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.98), eps=1e-9)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: self.lr_scheduler(self.embedding_dim, step, warmup=300))
        self.criterion = torch.nn.MSELoss()     
        self.epochs = 100
        self.patience = 5
        
    def lr_scheduler(self, dim_model: int, step:int, warmup:int):
        if step == 0:
            step = 1
        return (dim_model ** -0.5) * min(step ** -0.5, step * warmup ** -1.5)

   
    def train(
        self,
        training_dataloader:torch.utils.data.DataLoader,
        val_dataloader:torch.utils.data.DataLoader,
        batch_size: int = 8,
        patience: int = 5
    ) -> None:
        '''
        Parameters:
        input_audio: np.ndarray
            Input audio data, 3D array of shape (num_samples, audio_size, audio_val)
        output_imgs: np.ndarray
            Output image data, 3D array of shape (num_samples, img_size, img_val)
        '''
        self.model.to(self.device)
        self.criterion.to(self.device)
        
        for epoch in range(self.epochs):
            self.model.train()
            
            total_loss = 0
            print(f"== Epoch: {epoch}, Device: {self.device} ==")

            for img, spectrum in tqdm(training_dataloader):
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
            
            print(f"== Training Loss: {total_loss / len(train_dataloader)}, Device: {self.device}")
            
            # self.model.eval()
            
            # val_loss = 0
            # lowest_val_loss = float('inf')
            # wait_count = 0
            
            # with torch.no_grad():
            #     for i, (audio, img) in enumerate(val_dataloader):
            #         # Compare the predicted image with the actual image with some function
                    
            #         audio = audio.to(self.device)
            #         img = img.to(self.device)
            #         img = img.int()
                    
            #         gen_img = self.model.generate_image(audio)
            #         loss = self.validation_criterion(gen_img, img)
            #         val_loss += loss/val_dataloader.batch_size
                
            #     if val_loss < lowest_val_loss:
            #         lowest_val_loss = val_loss
            #         wait_count = 0
            #     else:
            #         wait_count += 1
            #         if wait_count == patience:

            #             print(f"Early Stopping at Epoch: {epoch}")
            #             break
            
            # print(f"== Validation Loss: {val_loss}, Device: {self.device}")
        
        print(f"Training Complete")
            

    def test(
        self,
        testing_dataloader:torch.utils.data.DataLoader
    ):
        self.model.to(self.device)
        self.criterion.to(self.device)
        
        self.model.eval()
        
        test_loss = 0
        
        with torch.no_grad():
            for audio, img in tqdm(testing_dataloader):
                audio = audio.to(self.device)
                img = img.to(self.device)
                img = img.int()
            
                gen_img = self.model.generate_image(audio)

                gen_img_np = gen_img.detach().cpu().numpy().astype(np.float32)
                img_np = img.detach().cpu().numpy().astype(np.float32) 

                loss = self.validation_criterion(gen_img_np, img_np, data_range=259.0)
                test_loss += loss/test_dataloader.batch_size
        
        print(f"Test Loss: {test_loss}, Device: {self.device}")             


if __name__ == "__main__":

    config = {
        'batch size': 5,
        'train ratio': 0.8,
        'validation ratio': 0.1,
        'test ratio': 0.1,
        'device': 'mps'
    }

    # Load the dataset
    ds_path = "datasets/AstroImg2Spec_ds_1000.pt"
    ds = torch.load(ds_path)
    
    # Split Train, Val, Test
    train_size = int(config['train ratio']*len(ds))
    # train_size = 4000
    val_size = int(config['validation ratio']*len(ds))
    test_size = len(ds) - train_size - val_size
    
    train, val, test = torch.utils.data.random_split(ds, [train_size, val_size, test_size])
    train = Subset(ds, range(5))
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=config['batch size'], shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=config['batch size'], shuffle=True)    
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=config['batch size'], shuffle=True)
    
    a2i_core = AstroImage2Spec(device=config['device'])
    
    # Chack size of model
    # total_params = sum(p.numel() for p in a2i_core.model.parameters())
    # print(f"Number of parameters: {total_params}")
    
    
    # Train
    a2i_core.train(train_dataloader, val_dataloader, batch_size=config['batch size'])
    
    # Test
    # a2i_core.test(test_dataloader)

    # Save the model
    model_path = "model/model.pt"
    torch.save(a2i_core.model, model_path)
