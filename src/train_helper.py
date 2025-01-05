from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

import torch


def train_ae(model, train_loader, optimizer, criterion, epochs, log_dir):
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
            for batch in pbar:
                inputs = batch[0]
                latent, outputs = model(inputs)
                loss = criterion(outputs, inputs)
                train_loss += loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                pbar.set_postfix(loss=loss.item())
        
        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)

    writer.close()

def train_multihead_ae(model, train_loader, optimizer, criterion, epochs, log_dir, weights):
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
            for batch in pbar:
                league_club, national_club, league_history, attributes, categorical = batch
                original_inputs = torch.cat([league_club, national_club, league_history, attributes], dim=1)

                latent, reconstruction = model(league_club, national_club, league_history, attributes, categorical)
                
                original_inputs = torch.cat([league_club, national_club, league_history, attributes, categorical], dim=1)
                
                loss = criterion(reconstruction, original_inputs, weights)
                train_loss += loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
                optimizer.step()
                
                pbar.set_postfix(loss=loss.item())
        
        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)

    writer.close()
