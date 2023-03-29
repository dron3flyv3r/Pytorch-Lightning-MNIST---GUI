import argparse
import tkinter
from tkinter.ttk import Progressbar
import random
import pytorch_lightning as pl
import torch.nn as nn
import torch
import torchvision
from PIL import ImageGrab, Image, ImageFilter, ImageOps

class Net(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(28*28, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(x.size(0), -1) 
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return {"loss": loss}
        
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)
    
    def train_dataloader(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomRotation(25),
            torchvision.transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3, fill=0),
            torchvision.transforms.RandomResizedCrop(28, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
            torchvision.transforms.CenterCrop(28),
            torchvision.transforms.ToTensor(),
        ])
        dataset = torchvision.datasets.MNIST(root=".", train=True, download=True, transform=transform)
        return torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=12)


class GUI:
    def __init__(self, model):
        # Create the root window
        self.root = tkinter.Tk()
        self.root.title("Hello World!")
        self.root.geometry("600x300")
        self.root.resizable(False, False)
        self.model = model
        
        # Create the progress bars
        self.progress_bars = []
        for i in range(10):
            progress = Progressbar(self.root, orient=tkinter.HORIZONTAL, length=200, mode='determinate')
            progress.grid(column=2, row=i)
            progress['value'] = 0
            progress['maximum'] = 100
            # Give it a label
            label = tkinter.Label(self.root, text=f"{i}")
            label.grid(column=1, row=i)
            self.progress_bars.append(progress)
        
        # Create a canvas for drawing of digit images and their probabilities
        self.canvas = tkinter.Canvas(self.root, width=280, height=280, bg="white", bd=0, highlightthickness=0)
        self.canvas.grid(column=0, row=0, rowspan=10)
        self.canvas.bind("<B1-Motion>", self.draw)
        # on release left click, predict digit  
        self.canvas.bind("<ButtonRelease-1>", self.predict_digit)
        self.canvas.bind("<Button-3>", lambda event: self.clear())
        self.canvas.bind("<BackSpace>", lambda event: self.clear())
        self.root.mainloop()
        
    def draw(self, event):
        x, y = event.x, event.y
        r = 12
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="black", outline="black")       
         
    def clear(self):
        self.canvas.delete("all")
        
    def predict_digit(self, event):
        # move the model to the cpu
        self.model.cpu()
        
        # Get the image from the canvas
        x = self.root.winfo_rootx() + self.canvas.winfo_x()
        y = self.root.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        image = ImageGrab.grab().crop((x, y, x1, y1))
        # save the image for debugging
        image = image.filter(ImageFilter.GaussianBlur(radius=3))
        
        image.save("image.png")
        image = ImageOps.invert(image)
        # Convert to grayscale and resize to 28x28
        image = image.convert("L")
        image = image.resize((28, 28), Image.ANTIALIAS) 
        # save the image for debugging
        image.save("image2.png")
        # Convert to tensor
        image = torchvision.transforms.ToTensor()(image)
        
        # Predict the digit
        with torch.no_grad():
            image = image.cpu()
            y_hat = self.model(image)
            y_hat = nn.functional.softmax(y_hat, dim=1)
            y_hat = y_hat[0]
            # Update the progress bars
            for i, progress in enumerate(self.progress_bars):
                progress['value'] = y_hat[i].item() * 100
    
if __name__ == "__main__":
    # Parse the arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", "-t", action="store_true")
    
    if parser.parse_args().train:
        model = Net()
        trainer = pl.Trainer(max_epochs=150, reload_dataloaders_every_n_epochs=50)
        trainer.fit(model)
        trainer.save_checkpoint("model.ckpt")
    else:
        model = Net.load_from_checkpoint("model.ckpt")
    GUI(model)