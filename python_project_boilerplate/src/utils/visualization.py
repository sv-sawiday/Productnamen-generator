import matplotlib.pyplot as plt
import seaborn as sns

def plot_distribution(data, column):
    sns.histplot(data[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.show()