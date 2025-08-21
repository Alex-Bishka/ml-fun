import pickle
import pandas as pd
import plotly.express as px
    
def load_loss_dict(file_path):
    with open(file_path, "rb") as f:
        loss_dict = pickle.load(f)

    return loss_dict


def create_df(merged_loss_dict):
    df = pd.DataFrame.from_dict(merged_loss_dict, orient="index").reset_index()
    df = df.rename(columns={'index': "Loss_Factor"})

    return df

def plot_loss_factors(df):
    fig = px.line(
        df,
        x='Loss_Factor',
        y='Final_Accuracy',
        title='Accuracy vs. Loss Factor',
        markers=True  # Add markers to the lines
    )
    
    fig.write_json("temp.json")
    
    # Show the plot
    fig.show()

def main(loss_dict_paths):
    loss_dict_iter = {}
    for path in loss_dict_paths:
        temp_d = load_loss_dict(path)
        loss_dict_iter = loss_dict_iter | temp_d
    
    df = create_df(loss_dict_iter)
    print(df.sort_values(by='Final_Accuracy', ascending=False).head())
    plot_loss_factors(df)

    return df