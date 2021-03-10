import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import Tk
import torch
from torch import nn


#umodel = torch.load('model0.147-0.840')
#lowermodel = torch.load('model0.061-0.864')
better_lowermodel=torch.load('model0.047-0.887')
model=better_lowermodel
class SampleApp(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        Tk.geometry(self, '690x600')

        self.frames = {}
        # Creating a loop to load all of the pages when buttons are pressed.
        for F in (StartPage, PageOne):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        print(cont.__name__)
        frame.tkraise()
        SampleApp.configure(self, bg="#4f4848")

class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        intro_msg = '''Hello! Welcome to Verify-19! This is the source of REAL Covid-19 News Validation! Click on our URL checker and validate your news in a flash!
        We use state of the art deep neural networks that makes a prediction on if your news is real or fake. What are you waiting for? Verify away!'''
        intro = tk.Label(self, text=intro_msg, wraplength=550)
        intro.pack()

        button = ttk.Button(self, text="Url Checker",
                            command=lambda: controller.show_frame(PageOne))
        button.pack()

        StartPage.configure(self, bg="#4f4848")


class PageOne(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        button = ttk.Button(self, text="Back to Home Screen",
                            command=lambda: controller.show_frame(StartPage))
        button.pack()
        self.title = ("Verify-19")
        self.entry = tk.Text(self, width=60, height=15, relief=GROOVE, borderwidth=3)
        self.clear_button = tk.Button(self, text='clear', command=self.clear)
        self.entry.pack(anchor=tk.CENTER)
        self.button = tk.Button(self, text="Submit", command=self.on_button, width=10)
        self.button.pack(anchor=CENTER, pady=4)
        self.clear_button.pack(pady=3)
        self.var = IntVar()
        self.show_var = IntVar()
        self.result_label = Label(self, text='', bg='#4f4848')
        self.status = Label(self, text='awaiting input')
        self.status.pack()
        self.result_label.pack(side=TOP)
        self.query = None
        self.user_input = None
        PageOne.configure(self, bg="#4f4848")

    def clear(self):
        self.entry.delete(1., 'end')

    def on_button(self):
        user_input = self.entry.get('1.0', tk.END)
        display = True
        result=model(user_input)
        if display:
            self.result_label.config(text=f'Probability of being true: {str(float(result))}', fg='#d9ced0')

app=SampleApp()
app.mainloop()
