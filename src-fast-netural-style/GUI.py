import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

from main import stylize

def style_transfer_function(content_image_path, style,outputname="output.jpg"):
    stylized_image=stylize(content_img_path=content_image_path, model_path="./saved_model/"+style+".pth", output_name=outputname)
    stylized_image=Image.open("./images/output/"+outputname)
    return stylized_image

# 全局变量保存原始图片
original_image = None
original_image_path = None
def upload_image():
    global original_image, original_image_path,tk_image
    original_image_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if original_image_path:
        original_image = Image.open(original_image_path)
        # 调整图片尺寸以便在界面中显示
        display_image = original_image.resize((300, 300))
        tk_image = ImageTk.PhotoImage(display_image)
        left_image_label.config(image=tk_image)
        left_image_label.image = tk_image  # 保持引用

def apply_style():
    if original_image is None:
        return  # 如果没有上传图片，则直接返回
    if original_image_path is None:
        print("请先上传图片")
        return
    selected_style = style_var.get()
    if selected_style == "风格1":
        selected_style = "wave_model"
    elif selected_style == "风格2":
        selected_style = "monet_model"

    print("应用风格：", selected_style)
    print("原始图片:",original_image_path)
    # 调用风格迁移函数处理图片
    styled_image = style_transfer_function(original_image_path, selected_style)
    display_styled = styled_image.resize((300, 300))
    styled_tk = ImageTk.PhotoImage(display_styled)
    right_image_label.config(image=styled_tk)
    right_image_label.image = styled_tk  # 保持引用

# 创建主窗口
root = tk.Tk()
root.title("风格迁移程序")
root.geometry("900x600")  # 设置窗口初始大小

# 顶部区域：上传图片和风格选择区域
top_frame = tk.Frame(root)
top_frame.pack(pady=10)

upload_button = tk.Button(top_frame, text="上传图片", command=upload_image,
                          width=20, height=2, font=("Arial", 14))
upload_button.pack(side=tk.LEFT, padx=10)

style_var = tk.StringVar(root)
style_var.set("风格1")
style_options = ["风格1", "风格2", "风格3"]
style_menu = tk.OptionMenu(top_frame, style_var, *style_options)
style_menu.config(font=("Arial", 12), width=10)
style_menu.pack(side=tk.LEFT, padx=10)

apply_button = tk.Button(top_frame, text="应用风格", command=apply_style,
                         width=20, height=2, font=("Arial", 14))
apply_button.pack(side=tk.LEFT, padx=10)

# 主区域：左右两个框架分别显示上传的图片和生成的新图片
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# 左侧框架用于显示上传的图片
left_frame = tk.Frame(main_frame)
left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

left_title = tk.Label(left_frame, text="上传图片", font=("Arial", 12))
left_title.pack(pady=5)
left_image_label = tk.Label(left_frame)
left_image_label.pack(pady=5)

# 右侧框架用于显示风格迁移后的图片
right_frame = tk.Frame(main_frame)
right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

right_title = tk.Label(right_frame, text="风格迁移后的图片", font=("Arial", 12))
right_title.pack(pady=5)
right_image_label = tk.Label(right_frame)
right_image_label.pack(pady=5)

root.mainloop()
