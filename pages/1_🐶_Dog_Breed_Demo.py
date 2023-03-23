import pandas as pd
import numpy as np

import streamlit as st
from PIL import Image
from io import BytesIO

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models

st.set_page_config(page_title="Dog Breed Demo", page_icon=":dog:")


STYLE = """
<style>
img {
    max-width: 100%;
}
</style>
"""

class ImageClassificationBase(nn.Module):
    # training step
    def training_step(self, batch):
        img, targets = batch
        out = self(img)
        loss = F.nll_loss(out, targets)
        return loss
    
    # validation step
    def validation_step(self, batch):
        img, targets = batch
        out = self(img)
        loss = F.nll_loss(out, targets)
        acc = accuracy(out, targets)
        return {'val_acc':acc.detach(), 'val_loss':loss.detach()}
    
    # validation epoch end
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss':epoch_loss.item(), 'val_acc':epoch_acc.item()}
        
    # print result end epoch
    def epoch_end(self, epoch, result):
        print("Epoch [{}] : train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result["train_loss"], result["val_loss"], result["val_acc"]))
    
        
class DogBreedPretrainedWideResnet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        
        self.network = models.wide_resnet50_2(pretrained=True)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Sequential(
            nn.Linear(num_ftrs, 120),
            nn.LogSoftmax(dim=1)
        )
        
    def forward(self, xb):
        return self.network(xb)
    
class FileUpload(object):

    def __init__(self):
        self.fileTypes = ["jpg", "jpeg"]
        self.image = None

    def run(self):
        """
        Upload a file on streamlit
        """
        st.info(__doc__)
        st.markdown(STYLE, unsafe_allow_html=True)
        file = st.file_uploader("Upload file", type=self.fileTypes)
        show_file = st.empty()
        if not file:
            show_file.info("Please upload a file of type: " + ", ".join(["jpg ", "jpeg"]))
            return
        content = file.getvalue()
        if isinstance(file, BytesIO):
            show_file.image(file)
            self.image = np.array(Image.open(file))
        file.close()

if __name__ == "__main__":
    st.header("What's this dog?")
    line_1 = "I have 'deployed' a model here for demonstration purposes. It's a ResNet model that can classify dog breeds from an image. "
    line_2 = "The model has been trained on the famous Stanford Dogs dataset, using transfer learning in PyTorch. "
    line_3 = "Have a try below. :)"
    st.markdown(line_1 + line_2 + line_3)

    # load model
    PATH = "assets/dog-breed-classifier-wideresnet_with_data_aug.pth"
    model = DogBreedPretrainedWideResnet()
    model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
    model.eval()

    # load list of breeds
    breeds = pd.read_csv('assets/breeds.csv')

    # define transformers
    resize_transform = transforms.Compose([  
        transforms.ToTensor(),
        transforms.Resize((168,168)) 
    ])

    helper = FileUpload()
    helper.run()

    if type(helper.image).__module__ == np.__name__:
        # image = cv2.resize(helper.image, (168, 168))
        image = resize_transform(helper.image)
        output = model(image.unsqueeze(0))
        prediction = output[0]
        index = torch.max(prediction, dim=0)[-1].item()
        label = breeds.loc[index].values[0]
        st.text(F"PREDICTION: {label}")