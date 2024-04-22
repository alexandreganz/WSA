# WSA PoC
This repo contains all the code related to the WSA PoC.

## Getting Started

### 1. Data
Download the WSA dataset from the Teams channel and put its content in _data/WSA_. Afterwards, your data folder should look like this:

- /data
    - /input
    - /WSA
        - /02-Assets
        - /03-Resource
        - 01-How to charge Pure Charge&Go AX with Pure Dry&Clean _ Signia Hearing Aids.mp4

### 2. Create Python Environment
Create new virtual environment
```
python -m venv .venv
```

Activate virtual environment
```
.\.venv\Scripts\activate
```

Install all packages from requirements.txt file
```
pip install -r requirements.txt
```

## Jupyter Notebooks and VS Code
You need to select the Python venv in as your kernel. Here's what to do if you can't find it:
- Select the Python interpreter in VS Code with Ctrl+Shift+P
- Select Python:Select Interpreter
- Browse path and select /.venv/Scripts/python.exe
- Go to the Jupyter Notebook and click on Select Kernel in the top right corner
    - Select Another Kernel... -> Python Environments... -> There is your venv

(Optional) Create a Jupyter Notebook kernel to run Jupyter Notebook commands inside the virtual environment
```
ipython kernel install --user --name=<kernel_name>
```

If you created a Jupyter Notebook  kernel you need to restart VS Code, otherwise you won't be able to select the kernel.