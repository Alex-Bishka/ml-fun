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

    # for scaling
    df['S1_Probe_Acccuracy'] = df['S1_Probe_Acccuracy'] * 100
    df['S2_Probe_Acccuracy'] = df['S2_Probe_Acccuracy'] * 100

    df_long = pd.melt(
        df,
        id_vars=['Loss_Factor'],
        value_vars=['Final_Accuracy', 'S1_Probe_Acccuracy', 'S2_Probe_Acccuracy'],
        var_name='Metric',
        value_name='Value'
    )

    return df_long

def plot_loss_factors(df):
    fig = px.line(
        df,
        x='Loss_Factor',
        y='Value',
        color='Metric',
        title='Accuracy Metrics vs. Dictionary Keys',
        labels={'Loss_Factor': 'Loss_Factors', 'Value': 'Accuracy', 'Metric': 'Metric'},
        markers=True  # Add markers to the lines
    )
    
    # Update layout for better visualization
    fig.update_layout(
        yaxis=dict(range=[min(df['Value']) - 5, max(100, max(df['Value']) + 5)]),
        xaxis=dict(tickmode='linear'),  # Adjust if keys are non-numerical
        showlegend=True
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
    print(df.query("Metric=='S1_Probe_Acccuracy'").sort_values(by='Value', ascending=False).head())
    plot_loss_factors(df)

    return df