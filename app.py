import tkinter as tk
import customtkinter as ctk

from PIL import Image, ImageTk
from authtoken import auth_token  # Replace with your authentication token

import torch
from diffusers import StableDiffusionPipeline

# Initialize the app
app = tk.Tk()
app.geometry("532x632")
app.title("Stable Bud")
ctk.set_appearance_mode("dark")

# Input prompt field
prompt = ctk.CTkEntry(
    master=app,
    height=40,
    width=512,
    font=("Arial", 20),
    text_color="black",
    fg_color="white"
)
prompt.place(x=10, y=10)

# Image display label
lmain = ctk.CTkLabel(master=app, height=512, width=512)
lmain.place(x=10, y=110)

# Stable Diffusion setup
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    use_auth_token=auth_token
)

# Move to appropriate device
if device == "cpu":
    pipe.to("cpu")
else:
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)


def generate():
    try:
        # Generate the image
        image = pipe(prompt.get(), guidance_scale=8.5).images[0]
        image.save("generatedimage.png")
        
        # Display the image
        img = ImageTk.PhotoImage(image=image)
        lmain.configure(image=img)
        lmain.image = img  # Prevent garbage collection
    except Exception as e:
        print(f"Error: {e}")


# Generate button
trigger = ctk.CTkButton(
    master=app,
    height=40,
    width=120,
    font=("Arial", 20),
    text_color="white",
    fg_color="blue",
    command=generate,
)
trigger.configure(text="Generate")
trigger.place(x=206, y=60)

app.mainloop()
