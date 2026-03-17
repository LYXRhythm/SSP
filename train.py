import torch
import time
import copy
import logging
from pathlib import Path
import numpy as np
import random as rn
import torch.optim as optim
import torch.nn.functional as F
from model import CMNN_Compat, Embedding
from losses import SSPLoss
from load_data import get_loader
from evaluate import fx_calc_map_multilabel
from utils import get_training_args

def to_seed(seed=0):
    """Fix random seeds to ensure experimental reproducibility"""
    np.random.seed(seed)
    rn.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def setup_logging(args):
    """Initialize logging system"""
    log_dir = Path("results")
    log_dir.mkdir(exist_ok=True)
    
    log_path = log_dir / (
        f"{args.dataset}_"
        f"PartialRatio_{args.partial_ratio}_Lamda_{args.lamda}lamda_log.txt"
    )
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_path, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return log_path

def evaluate_on_test_set(model, emb, input_data_par, device, epoch):
    """Evaluate mAP on test set"""
    model.eval()
    emb.eval()
    
    with torch.no_grad():
        img_test = torch.tensor(input_data_par['img_test']).to(device)
        text_test = torch.tensor(input_data_par['text_test']).to(device)
        
        view1_feature, view2_feature = model(img_test, text_test)
        
        label = input_data_par['label_test']
        view1_feature = view1_feature.detach().cpu().numpy()
        view2_feature = view2_feature.detach().cpu().numpy()

        img_to_txt = fx_calc_map_multilabel(view1_feature, view2_feature, label, metric='cosine')
        txt_to_img = fx_calc_map_multilabel(view2_feature, view1_feature, label, metric='cosine')
        avg_map = (img_to_txt + txt_to_img) / 2.0 if (img_to_txt + txt_to_img) > 0 else 0.0
        
        logging.info(f"\n[Test Set Evaluation - Epoch {epoch}]")
        logging.info(f"    - Image to Text MAP = {img_to_txt:.6f}")
        logging.info(f"    - Text to Image MAP = {txt_to_img:.6f}")
        logging.info(f"    - Average MAP = {avg_map:.6f}")
    
    model.train()
    emb.train()
    return avg_map

def train_model(model, emb, data_loaders, input_data_par, optimizer, configs, device):
    time_start = time.time()
    
    # Initialize loss function
    criterion = SSPLoss(
        partial_labels=input_data_par['label_train'],  
        ema_decay=configs.ema_decay,
        img_partial_labels=input_data_par.get('img_partial_label'),  
        txt_partial_labels=input_data_par.get('txt_partial_label')   
    ).cuda()
    
    # Training history records
    mAP_history = []
    epoch_loss_history = []
    best_avg_map = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    for epoch in range(configs.MAX_EPOCH):
        logging.info('\nEpoch {}/{}'.format(epoch, configs.MAX_EPOCH))
        logging.info('-' * 25)
        
        for phase in ['train', 'valid']:
            running_loss = 0.0
            model.train() if phase == 'train' else model.eval()
            
            for batch_data in data_loaders[phase]:
                imgs, txts, img_labels, txt_labels, ori_labels, index = batch_data
                
                if torch.isnan(imgs).any() or torch.isnan(txts).any():
                    logging.warning(f"Epoch {epoch} {phase}: Skipping batch with NaN values")
                    continue
                
                batch_size = imgs.size(0)
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    if torch.cuda.is_available():
                        imgs = imgs.cuda()
                        txts = txts.cuda()
                        img_labels = img_labels.cuda()
                        txt_labels = txt_labels.cuda()
                        ori_labels = ori_labels.cuda()
                        index = index.cuda()
                    
                    W = emb(torch.eye(configs.data_class).cuda())
                    view1_feature, view2_feature = model(imgs, txts)
                    
                    view1_flat = view1_feature.view(view1_feature.shape[0], -1)
                    view2_flat = view2_feature.view(view2_feature.shape[0], -1)
                    view1_predict = F.softmax(view1_flat.mm(W.T), dim=1)
                    view2_predict = F.softmax(view2_flat.mm(W.T), dim=1)
                    
                    loss = criterion(
                        pred_img=view1_predict,      
                        pred_txt=view2_predict,      
                        sample_index=index,              
                        img_feat=view1_feature,      
                        txt_feat=view2_feature,      
                        configs=configs,
                        epoch=epoch        
                    )
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * batch_size
            
            dataset_size = len(data_loaders[phase].dataset)
            epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
            
            if phase == 'train':
                model.eval()
                t_imgs, t_txts, t_labels = [], [], []
                with torch.no_grad():
                    for batch_data in data_loaders['valid']:
                        imgs, txts, _, _, ori_labels, _ = batch_data
                        if torch.cuda.is_available():
                            imgs = imgs.cuda()
                            txts = txts.cuda()
                        
                        t_view1_feature, t_view2_feature = model(imgs, txts)
                        t_imgs.append(t_view1_feature.cpu().numpy())
                        t_txts.append(t_view2_feature.cpu().numpy())
                        t_labels.append(ori_labels.cpu().numpy())
                
                t_imgs = np.concatenate(t_imgs) if t_imgs else np.array([])
                t_txts = np.concatenate(t_txts) if t_txts else np.array([])
                t_labels = np.concatenate(t_labels) if t_labels else np.array([])
                
                img2txt = fx_calc_map_multilabel(t_imgs, t_txts, t_labels, metric='cosine') if len(t_imgs) > 0 else 0.0
                txt2img = fx_calc_map_multilabel(t_txts, t_imgs, t_labels, metric='cosine') if len(t_txts) > 0 else 0.0
                avg_map = (img2txt + txt2img) / 2. if (img2txt + txt2img) > 0 else 0.0
                mAP_history.append(avg_map)
                model.train()
            
            lr = optimizer.param_groups[0]['lr'] if optimizer.param_groups else 0.0
            if phase == 'train':
                logging.info(f"    - [{phase:<5}] Loss: {epoch_loss:>6.4f}  Img2Txt: {img2txt:>6.4f}  Txt2Img: {txt2img:>6.4f}  Lr: {lr:>6g}")
            else:
                logging.info(f"    - [{phase:<5}] Loss: {epoch_loss:>6.4f}  Avg mAP: {avg_map:>6.4f}")
                epoch_loss_history.append(epoch_loss)
            
            # Save the best model weights based on validation average mAP
            if phase == 'valid' and avg_map > best_avg_map:
                best_avg_map = avg_map
                best_model_wts = copy.deepcopy(model.state_dict())
        
        # Evaluate on test set every 10 epochs
        if (epoch + 1) % 10 == 0:
            evaluate_on_test_set(model, emb, input_data_par, device, epoch + 1)

    time_end = time.time()
    time_used = time_end - time_start
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(time_used // 60, time_used % 60))
    logging.info(f'Best validation Average mAP: {best_avg_map:.6f}')
    
    # Load the best model weights
    model.load_state_dict(best_model_wts)
    
    return model, mAP_history

def main():
    """Main function: initialize parameters, data, model and start training"""
    args = get_training_args()
    setup_logging(args)
    logging.info(args)
    
    # Device initialization
    device = torch.device(f"cuda:{args.GPU}" if torch.cuda.is_available() else "cpu")
    to_seed(args.seed)
    
    logging.info("\n[SSP]: Data loading starts...")
    dataset = args.dataset
    data_loader, input_data_par = get_loader(dataset, args.batch_size, args.partial_ratio)
    args.data_class = input_data_par['num_class']
    logging.info('    - Train Numbers: {train:>4}  Valid Numbers: {valid:>4}  Test Numbers: {test:>4}  Classes Numbers: {classes:>4}'.format(
                 train=input_data_par["img_train"].shape[0], 
                 valid=input_data_par["img_valid"].shape[0], 
                 test=input_data_par["img_test"].shape[0], 
                 classes=input_data_par["label_train"].shape[1]))
    
    # Model initialization
    model_ft = CMNN_Compat(
        img_input_dim=input_data_par['img_dim'], 
        text_input_dim=input_data_par['text_dim'], 
        output_dim=args.output_dim, 
        num_class=input_data_par['num_class']
    ).to(device)

    # Embedding layer and optimizer initialization
    emb = Embedding(args.data_class, args.output_dim).cuda()
    optimizer = optim.Adam([
        {'params': emb.parameters(), 'lr': args.lr},
        {'params': model_ft.parameters(), 'lr': args.lr}
    ])
    
    logging.info("\n[SSP]: Training starts...")
    model_ft, _ = train_model(model_ft, emb, data_loader, input_data_par, optimizer, args, device)

    logging.info("\n[Final Test Set Evaluation]")
    view1_feature, view2_feature = model_ft(
        torch.tensor(input_data_par['img_test']).to(device), 
        torch.tensor(input_data_par['text_test']).to(device)
    )
    
    label = input_data_par['label_test']
    view1_feature = view1_feature.detach().cpu().numpy()
    view2_feature = view2_feature.detach().cpu().numpy()

    img_to_txt = fx_calc_map_multilabel(view1_feature, view2_feature, label, metric='cosine')
    txt_to_img = fx_calc_map_multilabel(view2_feature, view1_feature, label, metric='cosine')
    avg_map = (img_to_txt + txt_to_img) / 2.0
    
    logging.info("\n[SSP FINAL RESULT]:")
    logging.info(f"    - Image to Text MAP = {img_to_txt:.6f}")
    logging.info(f"    - Text to Image MAP = {txt_to_img:.6f}")
    logging.info(f"    - Average MAP = {avg_map:.6f}")
    
if __name__ == '__main__':
    main()