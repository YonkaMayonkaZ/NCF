import re
import pandas as pd
import matplotlib.pyplot as plt

def parse_results(file_path):
    results = []
    factor_values = [8, 16, 32, 64, 128]
    models = ['GMF', 'MLP']
    
    factor_index = 0
    model_index = 0
    
    with open(file_path, 'r') as file:
        log_data = file.readlines()
    
    for line in log_data:
        match = re.search(r'End\. Best epoch (\d+): HR = ([0-9.]+), NDCG = ([0-9.]+)', line)
        if match:
            best_epoch = int(match.group(1))
            hr = float(match.group(2))
            ndcg = float(match.group(3))
            
            results.append({
                'Factor': factor_values[factor_index],
                'Model': models[model_index],
                'Best Epoch': best_epoch,
                'HR': hr,
                'NDCG': ndcg
            })
            
            model_index = (model_index + 1) % 2
            if model_index == 0:
                factor_index += 1
    
    return pd.DataFrame(results)

def plot_results(df):
    plt.figure(figsize=(10, 5))
    
    for model in df['Model'].unique():
        subset = df[df['Model'] == model]
        plt.plot(subset['Factor'], subset['HR'], marker='o', label=f'{model} - HR')
        plt.plot(subset['Factor'], subset['NDCG'], marker='s', label=f'{model} - NDCG')
    
    plt.xscale('log', base=2)
    plt.xlabel('Factor')
    plt.ylabel('Score')
    plt.title('Model Performance vs. Factor')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
file_path = "results.txt"
df = parse_results(file_path)
print(df)
plot_results(df)

