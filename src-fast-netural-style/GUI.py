import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

from main import stylize

# 全局变量保存原始图片和生成图片
original_image = None
original_image_path = None
output_image_path = "output.jpg"

def style_transfer_function(content_image_path, style):
    stylized_image = stylize(content_img_path=content_image_path, model_path="./src-fast-netural-style/saved_model/"+style, output_name=output_image_path)
    stylized_image = Image.open("./src-fast-netural-style/images/output/"+output_image_path)
    return stylized_image

def upload_image():
    global original_image, original_image_path, tk_image
    original_image_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if original_image_path:
        original_image = Image.open(original_image_path)
        display_image = original_image.resize((300, 300))
        tk_image = ImageTk.PhotoImage(display_image)
        left_image_label.config(image=tk_image)
        left_image_label.image = tk_image

def apply_style():
    if original_image is None:
        return
    if original_image_path is None:
        print("请先上传图片")
        return
    selected_style = style_var.get()
    if selected_style == "莫奈风格":
        selected_style = "monet.pth"
    elif selected_style == "糖果风格":
        selected_style = "candy.pth"
    elif selected_style == "海浪风格":
        selected_style = "wave_model.pth"

    print("应用风格：", selected_style)
    print("原始图片:", original_image_path)
    styled_image = style_transfer_function(original_image_path, selected_style)
    display_styled = styled_image.resize((300, 300))
    styled_tk = ImageTk.PhotoImage(display_styled)
    right_image_label.config(image=styled_tk)
    right_image_label.image = styled_tk

def start_application():
    start_frame.destroy()
    main_application()

def main_application():
    global left_image_label, right_image_label, style_var
    
    top_frame = tk.Frame(root)
    top_frame.pack(pady=10)

    upload_button = tk.Button(top_frame, text="上传图片", command=upload_image, width=20, height=2, font=("Arial", 14))
    upload_button.pack(side=tk.LEFT, padx=10)

    style_var = tk.StringVar(root)
    style_var.set("莫奈风格")
    style_options = ["莫奈风格", "糖果风格", "海浪风格"]
    style_menu = tk.OptionMenu(top_frame, style_var, *style_options)
    style_menu.config(font=("Arial", 12), width=10)
    style_menu.pack(side=tk.LEFT, padx=10)

    apply_button = tk.Button(top_frame, text="应用风格", command=apply_style, width=20, height=2, font=("Arial", 14))
    apply_button.pack(side=tk.LEFT, padx=10)

    main_frame = tk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    left_frame = tk.Frame(main_frame)
    left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    left_title = tk.Label(left_frame, text="上传图片", font=("Arial", 12))
    left_title.pack(pady=5)
    left_image_label = tk.Label(left_frame)
    left_image_label.pack(pady=5)

    right_frame = tk.Frame(main_frame)
    right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    right_title = tk.Label(right_frame, text="风格迁移后的图片", font=("Arial", 12))
    right_title.pack(pady=5)
    right_image_label = tk.Label(right_frame)
    right_image_label.pack(pady=5)

    bottom_right_frame = tk.Frame(root)
    bottom_right_frame.pack(side=tk.BOTTOM, anchor=tk.SE, padx=20, pady=20)
    
    bottom_image = Image.open("GUI_supplements/DA.jpg") 
    bottom_image = bottom_image.resize((128, 103))
    bottom_tk_image = ImageTk.PhotoImage(bottom_image)
    bottom_image_label = tk.Label(bottom_right_frame, image=bottom_tk_image)
    bottom_image_label.image = bottom_tk_image
    bottom_image_label.pack()

    bottom_text_label = tk.Label(bottom_right_frame, text="人工智能导论", font=("Arial", 10))
    bottom_text_label.pack()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("风格迁移程序")
    root.geometry("900x600")
    
    start_frame = tk.Frame(root)
    start_frame.pack(fill=tk.BOTH, expand=True)
    
    canvas = tk.Canvas(start_frame, width=900, height=600)
    canvas.pack(fill=tk.BOTH, expand=True)
    
    start_image = Image.open("GUI_supplements/start.jpg")
    start_image = start_image.resize((256, 206))
    start_tk_image = ImageTk.PhotoImage(start_image)
    canvas.create_image(850, 550, anchor=tk.SE, image=start_tk_image)
    
    canvas.create_text(450, 250, text="风格迁移程序", font=("Arial", 40, "bold"), fill="red")
    canvas.create_text(450, 300, text="刘东琛 赖星宇 张皓哲", font=("Arial", 20, "bold"), fill="black")
    canvas.create_text(450, 550, text="git@github.com:Li-Battery-dc/netural-transfer.git", font=("Arial", 10, "bold"), fill="black")
    
    start_button = tk.Button(start_frame, text="开始", command=start_application, font=("Arial", 14), width=10, height=2)
    start_button_window = canvas.create_window(450, 400, window=start_button)
    
    root.mainloop()
