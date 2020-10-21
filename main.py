# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    cali_dataset = pd.read_csv("datasets/california_paper_eRCNN/I5-N-3/2015.csv")
    cali_dataset.head()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
