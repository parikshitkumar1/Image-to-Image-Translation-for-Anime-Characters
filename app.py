from p2p import *

import streamlit as st
from PIL import Image

st.markdown(
    """
    <style>
    .reportview-container {
        background: url("https://i.pinimg.com/originals/aa/dc/21/aadc21a521165193480e7f2b5f22c3d1.jpg")
    }
   .sidebar .sidebar-content {
        background: url("url_goes_here")
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("C2")
st.subheader("Pix2Pix GAN for Image-to-Image Translation")

@st.cache
def load_image(image_file):
	img = Image.open(image_file)
	return img

image_file = st.file_uploader("Upload An Image (jpg/jpeg)",type=['png','jpeg','jpg'])
if image_file is not None:
    file_details = {"FileName":image_file.name,"FileType":image_file.type}
    st.write(file_details)
    img = load_image(image_file)
    st.image(img, width = 256)
    with open(os.path.join("utils/images","image"),"wb") as f: 
      f.write(image_file.getbuffer())         
    st.success("Saved File")


gen = torch.load("utils/WEIGHTS/pix2pix.pth", map_location = "cpu")
DEVICE = "cpu"

dir = "utils/images"
val_dataset = AnimuDataset(root_dir=dir)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

x, y = next(iter(val_loader))
x, y = x.to(DEVICE), y.to(DEVICE)
gen.eval()
epoch = 1
folder = "utils/result"

with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization#
        save_image(y_fake, folder + f"/y_gen_{epoch}.png")


st.write("Pixelating...")
result = "utils/result/y_gen_1.png"
st.image(result)




