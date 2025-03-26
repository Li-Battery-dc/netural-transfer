import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

from src_fast_netural_style.main import stylize
from Wikiart_dataset_include_classifier.eval.use_model import classify
# 全局变量保存原始图片和生成图片
original_image = None
original_image_path = None
output_image_path = "output.jpg"


def style_transfer_function(content_image_path, style):
    stylized_image = stylize(content_img_path=content_image_path,
                               model_path="./src_fast_netural_style/saved_model/" + style,
                               output_dir="./src_fast_netural_style/images/output/",
                               output_name=output_image_path)
    stylized_image = Image.open("./src_fast_netural_style/images/output/" + output_image_path)
    return stylized_image

def upload_image():
    global original_image, original_image_path, tk_image
    original_image_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if original_image_path:
        original_image = Image.open(original_image_path)
        display_image = original_image.resize((300, 300))
        tk_image = ImageTk.PhotoImage(display_image)
        left_image_label.config(image=tk_image)
        left_image_label.image = tk_image

def apply_style():
    if original_image is None or original_image_path is None:
        print("请先上传图片")
        return
    selected_style = style_var.get()
    if selected_style == "莫奈风格":
        selected_style = "monet.pth"
    elif selected_style == "日出印象风格":
        selected_style = "monet_sunrise.pth"
    elif selected_style == "立体派风格":
        selected_style = "cubist.pth"
    elif selected_style == "海浪风格":
        selected_style = "wave.pth"
    elif selected_style == "羽毛风格":
        selected_style = "feathers.pth"
    elif selected_style == "毕加索风":
        selected_style = "picasso.pth"
    elif selected_style == "呐喊风格":
        selected_style = "scream.pth"
    elif selected_style == "梵高星空风格":
        selected_style = "starry_night.pth"
    elif selected_style == "udnie风格":
        selected_style = "udnie.pth"

    print("应用风格：", selected_style)
    print("原始图片:", original_image_path)
    styled_image = style_transfer_function(original_image_path, selected_style)
    display_styled = styled_image.resize((300, 300))
    styled_tk = ImageTk.PhotoImage(display_styled)
    right_image_label.config(image=styled_tk)
    right_image_label.image = styled_tk

def return_to_main():
    main_start()

def start_application():
    start_frame.destroy()
    main_application()

def start_classification():
    start_frame.destroy()
    main_classification()

def main_application():
    global left_image_label, right_image_label, style_var

    top_frame = tk.Frame(root)
    top_frame.pack(pady=10)

    upload_button = tk.Button(top_frame, text="上传图片", command=upload_image,
                              width=20, height=2, font=("Arial", 14))
    upload_button.pack(side=tk.LEFT, padx=10)

    style_var = tk.StringVar(root)
    style_var.set("莫奈风格")
    style_options = ["莫奈风格", "日出印象风格", "立体派风格", "海浪风格", "羽毛风格", "毕加索风", "呐喊风格", "梵高星空风格", "udnie风格"]
    style_menu = tk.OptionMenu(top_frame, style_var, *style_options)
    style_menu.config(font=("Arial", 12), width=10)
    style_menu.pack(side=tk.LEFT, padx=10)

    apply_button = tk.Button(top_frame, text="应用风格", command=apply_style,
                             width=20, height=2, font=("Arial", 14))
    apply_button.pack(side=tk.LEFT, padx=10)

    main_frame = tk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    left_frame = tk.Frame(main_frame)
    left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    left_title = tk.Label(left_frame, text="上传图片", font=("Arial", 12))
    left_title.pack(pady=5)
    global left_image_label
    left_image_label = tk.Label(left_frame)
    left_image_label.pack(pady=5)

    right_frame = tk.Frame(main_frame)
    right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    right_title = tk.Label(right_frame, text="风格迁移后的图片", font=("Arial", 12))
    right_title.pack(pady=5)
    global right_image_label
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
    
    # 返回按钮，放置在左下角
    return_button = tk.Button(root, text="返回", command=return_to_main,
                              font=("Arial", 12), width=8, height=1)
    return_button.place(relx=0.05, rely=0.95, anchor=tk.SW)

def upload_and_classify():
    global uploaded_image, uploaded_image_path, tk_uploaded_image
    uploaded_image_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if uploaded_image_path:
        uploaded_image = Image.open(uploaded_image_path)
        display_image = uploaded_image.resize((300, 300))
        tk_uploaded_image = ImageTk.PhotoImage(display_image)
        image_label.config(image=tk_uploaded_image)
        image_label.image = tk_uploaded_image
        
        # 调用 classify 进行分类，并显示结果
        category = classify(uploaded_image_path)
        result_label.config(text=f"分类结果: {category}")

def main_classification():
    global image_label, result_label
    for widget in root.winfo_children():
        widget.destroy()
    
    top_frame = tk.Frame(root)
    top_frame.pack(pady=10)
    
    upload_button = tk.Button(top_frame, text="上传图片", command=upload_and_classify,
                              width=20, height=2, font=("Arial", 14))
    upload_button.pack()

    main_frame = tk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    global image_label, result_label
    image_label = tk.Label(main_frame)
    image_label.pack(pady=5)
    
    result_label = tk.Label(main_frame, text="分类结果: ", font=("Arial", 12))
    result_label.pack(pady=10)
    
    # 返回按钮，放置在左下角
    return_button = tk.Button(root, text="返回", command=return_to_main,
                              font=("Arial", 12), width=8, height=1)
    return_button.place(relx=0.05, rely=0.95, anchor=tk.SW)

def main_start():
    global start_frame, canvas, start_tk_image
    for widget in root.winfo_children():
        widget.destroy()
    
    start_frame = tk.Frame(root)
    start_frame.pack(fill=tk.BOTH, expand=True)
    
    canvas = tk.Canvas(start_frame, width=900, height=600)
    canvas.pack(fill=tk.BOTH, expand=True)
    
    start_image = Image.open("GUI_supplements/start.jpg")
    start_image = start_image.resize((256, 206))
    start_tk_image = ImageTk.PhotoImage(start_image)
    # 保持对 start_tk_image 的引用，避免被垃圾回收
    canvas.image = start_tk_image
    canvas.create_image(850, 550, anchor=tk.SE, image=start_tk_image)
    
    canvas.create_text(450, 250, text="风格迁移程序", font=("Arial", 40, "bold"), fill="red")
    canvas.create_text(450, 300, text="刘东琛 赖星宇 张皓哲", font=("Arial", 20, "bold"), fill="black")
    canvas.create_text(450, 550, text="git@github.com:Li-Battery-dc/netural-transfer.git",
                       font=("Arial", 10, "bold"), fill="black")
    
    start1_button = tk.Button(start_frame, text="风格迁移", command=start_application,
                              font=("Arial", 14), width=10, height=2)
    canvas.create_window(450, 400, window=start1_button)

    start2_button = tk.Button(start_frame, text="分类", command=start_classification,
                              font=("Arial", 14), width=10, height=2)
    canvas.create_window(450, 470, window=start2_button)

def main():
    global root
    root = tk.Tk()
    root.title("风格迁移程序")
    root.geometry("900x600")
    main_start()
    root.mainloop()

if __name__ == "__main__":
    main()
