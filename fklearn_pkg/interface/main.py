from plot import *


# Load custom styles
custom_css = Div(text="<link rel='stylesheet' type='text/css' href='fklearn/interface/static/css/styles.css'>")
add_btn = Button(label="Add Plot", button_type="success")
remove_btn = Button(label="Remove Plot", button_type="danger")

# Construct our viewport
l = layout([
    [custom_css],
    create_plot(),
    [add_btn, remove_btn]
], sizing_mode="fixed", css_classes=["layout-container"])

def add_plot():
    l.children.insert(len(l.children)-1, create_plot())

def remove_plot():
    if len(l.children) > 3:
        l.children.pop(len(l.children)-2)

add_btn.on_click(add_plot)
remove_btn.on_click(remove_plot)

curdoc().add_root(l)
curdoc().title = "FKLEARN"
